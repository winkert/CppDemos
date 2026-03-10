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
            var model = new ModelProto
            {
                IrVersion = 8,
                ProducerName = "UnitTest",
                OpsetImport = { new OperatorSetIdProto { Version = 17 } }
            };

            var graph = new GraphProto { Name = "UnitTestGraph" };

            // Prepare a simple dense layer: input size = 2, output size = 3
            int inputSize = 2;
            int outputSize = 3;
            float[] weights = new float[inputSize * outputSize];
            float[] biases = new float[outputSize];

            // fill with deterministic values
            for (int i = 0; i < weights.Length; i++) weights[i] = 0.1f + i * 0.01f;
            for (int i = 0; i < biases.Length; i++) biases[i] = 0.2f + i * 0.01f;

            // Add initializers expected by ImportFromOnnx
            graph.Initializer.Add(new TensorProto
            {
                Name = "W0",
                DataType = (int)TensorProto.Types.DataType.Float,
            }.Also(t =>
            {
                t.Dims.Add(outputSize);
                t.Dims.Add(inputSize);
                t.FloatData.AddRange(weights);
            }));

            graph.Initializer.Add(new TensorProto
            {
                Name = "B0",
                DataType = (int)TensorProto.Types.DataType.Float,
            }.Also(t =>
            {
                t.Dims.Add(outputSize);
                t.FloatData.AddRange(biases);
            }));

            // Gemm node
            graph.Node.Add(new NodeProto
            {
                OpType = "Gemm",
                Input = { "input", "W0", "B0" },
                Output = { "GemmOut0" }
            });

            // Activation node (Relu) following Gemm
            graph.Node.Add(new NodeProto
            {
                OpType = "Relu",
                Input = { "GemmOut0" },
                Output = { "ActOut0" }
            });

            // Minimal input/output definitions (not used by ImportFromOnnx but good to include)
            graph.Input.Add(new ValueInfoProto { Name = "input" });
            graph.Output.Add(new ValueInfoProto { Name = "ActOut0" });

            model.Graph = graph;

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
    }

    // Small helper extension used above to allow adding FloatData/Dims inline without extra variables.
    internal static class ProtoExtensions
    {
        public static T Also<T>(this T obj, System.Action<T> act)
        {
            act(obj);
            return obj;
        }
    }
}
