using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using System.Linq;
using Google.Protobuf;
using Onnx;
using TRW.CommonLibraries.NeuralNetwork;
using TRW.UnitTesting;

namespace TRW.CommonLibraries.NeuralNetwork.Test
{
    [TestClass]
    public class OnnxerTests : UnitTestBase
    {
        [TestMethod]
        public void ExportToOnnx_CreatesValidOnnxFile()
        {
            // Arrange
            var nn = new NeuralNetwork();
            // build a small network using public API
            nn.AddLayer(2, 3, ActivationFunction.ReLU);
            nn.AddLayer(3, 1, ActivationFunction.Sigmoid);

            var onnxer = new Onnxer();
            string path = Path.Combine(UnitTestTempFolder, "export_test.onnx");

            // Act
            onnxer.ExportToOnnx(nn, path);

            // Assert - file created and parseable
            Assert.IsTrue(File.Exists(path), "ONNX file was not created.");

            var model = ModelProto.Parser.ParseFrom(File.ReadAllBytes(path));
            Assert.IsNotNull(model);
            Assert.IsTrue(model.Graph.Node.Count > 0, "Exported model should contain nodes.");
            Assert.IsTrue(model.Graph.Initializer.Count > 0, "Exported model should contain initializers for weights/biases.");
        }

        [TestMethod]
        public void ImportFromOnnx_ParsesGemmAndActivationNodes()
        {
            // Arrange - create a minimal ONNX model with one Gemm and one Relu
            var model = TestMocker.MockOnnxerModel_Basic();

            string path = Path.Combine(UnitTestTempFolder, "import_test.onnx");
            using (var fs = File.Create(path))
            {
                model.WriteTo(fs);
            }

            var onnxer = new Onnxer();

            // Act
            var net = onnxer.ImportFromOnnx(path);

            // Assert - import did not fail and returned a network with at least one layer
            Assert.IsNotNull(net, "Imported neural network should not be null.");
            Assert.IsTrue(net.Layers.Count > 0, "Imported network should contain at least one layer.");
        }

        [TestMethod]
        public void ExportTransformerModel()
        {
            NeuralNetwork nn = TestMocker.TrainTransformerNetwork();

            var onnxer = new Onnxer();
            string path = Path.Combine(UnitTestTempFolder, "export_test.onnx");

            // Act
            onnxer.ExportToOnnx(nn, path);

            // Assert - file created and parseable
            Assert.IsTrue(File.Exists(path), "ONNX file was not created.");

            var model = ModelProto.Parser.ParseFrom(File.ReadAllBytes(path));
            Assert.IsNotNull(model);
            Assert.IsTrue(model.Graph.Node.Count > 0, "Exported model should contain nodes.");
            Assert.IsTrue(model.Graph.Initializer.Count > 0, "Exported model should contain initializers for weights/biases.");

            // === Additional assertions to validate TransformerBlock export ===
            
            // 1. Verify presence of transformer-specific ONNX operators
            var nodeOps = model.Graph.Node.Select(n => n.OpType).ToList();
            
            // LayerNormLayer should be exported as InstanceNormalization
            Assert.IsTrue(nodeOps.Contains("InstanceNormalization"), 
                "Transformer should export LayerNormLayer as InstanceNormalization.");
            
            // MultiHeadAttentionLayer should generate MatMul and Softmax for attention computation
            Assert.IsTrue(nodeOps.Contains("MatMul"), 
                "Transformer should contain MatMul nodes for attention computation.");
            Assert.IsTrue(nodeOps.Contains("Softmax"), 
                "Transformer should contain Softmax for attention weights.");
            Assert.IsTrue(nodeOps.Contains("Transpose"), 
                "Transformer should contain Transpose for key matrix in attention.");
            
            // FeedForwardLayer should contain at least multiple Gemm nodes
            var gemmCount = nodeOps.Count(op => op == "Gemm");
            Assert.IsTrue(gemmCount >= 4, 
                "Transformer block should have multiple Gemm nodes for projections (expected >= 4, got " + gemmCount + ")");

            // 2. Verify transformer-specific initializers exist
            var initializerNames = model.Graph.Initializer.Select(i => i.Name).ToList();
            
            // Check for multi-head attention weight names (Q, K, V, O projections)
            bool hasAttnWeights = initializerNames.Any(name => name.Contains("W_Q")) ||
                                   initializerNames.Any(name => name.Contains("W_K")) ||
                                   initializerNames.Any(name => name.Contains("W_V")) ||
                                   initializerNames.Any(name => name.Contains("W_O"));
            Assert.IsTrue(hasAttnWeights, 
                "Transformer should export multi-head attention weight matrices (Q, K, V, O).");
            
            // 3. Verify node count is appropriate for transformer complexity
            // TransformerBlock typically has:
            // - 2 LayerNorm nodes
            // - 1 MHA with ~6-7 operations (4 Gemm for Q/K/V/O + Transpose + MatMul + Softmax + MatMul)
            // - FeedForwardLayer with 2+ Gemm and activations
            // - Total: 15+ nodes for a single transformer block
            Assert.IsTrue(model.Graph.Node.Count >= 10, 
                "Transformer export should have substantial node count (expected >= 10, got " + model.Graph.Node.Count + ")");

            // 4. Verify input/output shapes are preserved
            Assert.IsTrue(model.Graph.Input.Count > 0, "Model should have input definitions.");
            Assert.IsTrue(model.Graph.Output.Count > 0, "Model should have output definitions.");
            
            // 5. Verify all nodes have valid inputs and outputs
            foreach (var node in model.Graph.Node)
            {
                Assert.IsTrue(node.Input.Count > 0, 
                    $"Node '{node.OpType}' should have at least one input.");
                Assert.IsTrue(node.Output.Count > 0, 
                    $"Node '{node.OpType}' should have at least one output.");
            }

            // 6. Verify initializers have float data
            foreach (var init in model.Graph.Initializer)
            {
                Assert.IsTrue(init.FloatData.Count > 0 || init.RawData.Length > 0,
                    $"Initializer '{init.Name}' should contain data (FloatData or RawData).");
            }
        }

        [TestMethod]
        public void ImportTransformerModel_RoundTrip()
        {
            // Arrange - create and export a transformer network
            var originalNet = TestMocker.TrainTransformerNetwork();
            var onnxer = new Onnxer();
            string exportPath = Path.Combine(UnitTestTempFolder, "transformer_export.onnx");
            string importPath = Path.Combine(UnitTestTempFolder, "transformer_import.onnx");

            // Act - Export the transformer network
            onnxer.ExportToOnnx(originalNet, exportPath);
            Assert.IsTrue(File.Exists(exportPath), "Export failed: ONNX file was not created.");

            // Verify original export has transformer components
            var originalModel = ModelProto.Parser.ParseFrom(File.ReadAllBytes(exportPath));
            var originalNodeOps = originalModel.Graph.Node.Select(n => n.OpType).ToList();
            Assert.IsTrue(originalNodeOps.Contains("InstanceNormalization"), 
                "Original export should contain InstanceNormalization from LayerNormLayer.");

            // Import it back
            var importedNet = onnxer.ImportFromOnnx(exportPath);
            Assert.IsNotNull(importedNet, "Imported neural network should not be null.");
            Assert.IsTrue(importedNet.Layers.Count > 0, "Imported network should contain layers.");

            // Export the imported network to verify it's exportable
            onnxer.ExportToOnnx(importedNet, importPath);
            Assert.IsTrue(File.Exists(importPath), "Re-export failed: ONNX file was not created.");

            // Verify re-exported model is valid and contains computation nodes
            var importedModel = ModelProto.Parser.ParseFrom(File.ReadAllBytes(importPath));
            Assert.IsTrue(importedModel.Graph.Node.Count > 0, "Re-exported model should have nodes.");
            Assert.IsTrue(importedModel.Graph.Initializer.Count > 0, "Re-exported model should have initializers.");

            var importedNodeOps = importedModel.Graph.Node.Select(n => n.OpType).ToList();
            
            // Should have core computation nodes (Gemm or MatMul)
            Assert.IsTrue(importedNodeOps.Contains("Gemm") || importedNodeOps.Contains("MatMul"), 
                "Re-exported model should contain computation nodes.");

            // Should have activation nodes
            bool hasActivations = importedNodeOps.Any(op => IsActivationOp(op));
            Assert.IsTrue(hasActivations, "Re-exported model should contain activation nodes.");
        }

        [TestMethod]
        public void ImportTransformerModel_DetectsLayerNormalization()
        {
            // Arrange - create a simple model with just InstanceNormalization
            var model = new ModelProto
            {
                IrVersion = 8,
                OpsetImport = { new OperatorSetIdProto { Version = 17 } },
                ProducerName = "TestOnnxer"
            };

            var graph = new GraphProto { Name = "TestGraph" };

            // Create input
            graph.Input.Add(new ValueInfoProto
            {
                Name = "input",
                Type = CreateTensorType(TensorProto.Types.DataType.Float, [1, 8])
            });

            // Create InstanceNormalization node (representing LayerNormLayer)
            graph.Node.Add(new NodeProto
            {
                OpType = "InstanceNormalization",
                Input = { "input" },
                Output = { "norm_out" },
                Attribute = { new AttributeProto { Name = "epsilon", F = 1e-5f } }
            });

            // Create output
            graph.Output.Add(new ValueInfoProto
            {
                Name = "norm_out",
                Type = CreateTensorType(TensorProto.Types.DataType.Float, [1, 8])
            });

            model.Graph = graph;

            string path = Path.Combine(UnitTestTempFolder, "layernorm_test.onnx");
            using (var fs = File.Create(path))
            {
                model.WriteTo(fs);
            }

            var onnxer = new Onnxer();

            // Act
            var net = onnxer.ImportFromOnnx(path);

            // Assert
            Assert.IsNotNull(net, "Imported network should not be null.");
            Assert.IsTrue(net.Layers.Count > 0, "Imported network should have at least one layer.");
            Assert.IsTrue(net.Layers[0] is LayerNormLayer, "First layer should be LayerNormLayer.");
        }

        private static TypeProto CreateTensorType(TensorProto.Types.DataType type, long[] dims)
        {
            var tp = new TypeProto();
            var tt = new TypeProto.Types.Tensor
            {
                ElemType = (int)type,
                Shape = new TensorShapeProto()
            };
            foreach (var d in dims)
                tt.Shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimValue = d });
            tp.TensorType = tt;
            return tp;
        }

        private static bool IsActivationOp(string opType) =>
            opType is "Relu" or "LeakyRelu" or "Tanh" or "Sigmoid" or "Softmax" or "Swish" or "HardSwish" or "ELU" or "GELU" or "InstanceNormalization";
    }

    
}
