using Google.Protobuf;
using Onnx;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection.Metadata.Ecma335;
using System.Text;
using System.Threading.Tasks;

namespace TRW.CommonLibraries.NeuralNetwork
{
    public class Onnxer
    {
        public void ExportToOnnx(NeuralNetwork network, string path)
        {
            var model = new ModelProto
            {
                IrVersion = 8,
                OpsetImport = { new OperatorSetIdProto { Version = 17 } },
                ProducerName = "HomebrewNN"
            };

            GraphProto graph = new GraphProto { Name = "HomebrewGraph" };

            string prevOutput = "input";

            // Define model input (you can adjust shape)
            graph.Input.Add(new ValueInfoProto
            {
                Name = "input",
                Type = MakeTensorTypeProto(Onnx.TensorProto.Types.DataType.Float, [1, network.InputSize])
            });

            // Use a global counter to ensure unique node names across nested layers
            int globalNodeIndex = 0;

            for (int i = 0; i < network.Layers.Count; i++)
            {
                ILayer layer = (LayerBase)network.Layers[i];
                if (layer is CompositeLayer)
                {
                    prevOutput = ExportCompositeLayer(graph, prevOutput, (CompositeLayer)layer, ref globalNodeIndex);
                }
                else
                {
                    prevOutput = ExportBasicLayer(graph, prevOutput, ref globalNodeIndex, layer);
                }
            }

            // Define model output
            graph.Output.Add(new ValueInfoProto
            {
                Name = prevOutput,
                Type = MakeTensorTypeProto(Onnx.TensorProto.Types.DataType.Float, [1, network.OutputSize])
            });

            model.Graph = graph;

            using var f = File.Create(path);
            model.WriteTo(f);
        }

        public NeuralNetwork ImportFromOnnx(string path)
        {
            var model = ModelProto.Parser.ParseFrom(File.ReadAllBytes(path));
            var graph = model.Graph;

            var net = new NeuralNetwork();
            var initMap = graph.Initializer.ToDictionary(i => i.Name);

            int i = 0;
            while (i < graph.Node.Count)
            {
                var node = graph.Node[i];
                double[] W, B;
                int outputSize, inputSize;

                // Case 1: LayerNormalization (from TransformerBlock)
                if (node.OpType == "InstanceNormalization")
                {
                    // LayerNormLayer has no trainable parameters
                    // Extract dimension from input or output tensor info
                    int dim = 1;
                    if (node.Input.Count > 0 && initMap.TryGetValue(node.Input[0], out var inputTensor) && inputTensor.Dims.Count > 0)
                    {
                        dim = (int)inputTensor.Dims[inputTensor.Dims.Count - 1];
                    }
                    
                    var normLayer = new LayerNormLayer(dim);
                    net.AddLayer(normLayer);
                    i++;
                }
                // Case 2: MultiHeadAttentionLayer pattern detection
                else if (TryParseMultiHeadAttention(graph, i, initMap, out var mhaLayer, out var nodesConsumed))
                {
                    net.AddLayer(mhaLayer);
                    i += nodesConsumed;
                }
                // Case 3: Dense layer (Gemm)
                else if (node.OpType == "Gemm")
                {
                    GetNodeAttributes(initMap, node, out W, out B, out outputSize, out inputSize);

                    var dense = new DenseLayer(inputSize, outputSize)
                    {
                        Weights = W,
                        Biases = B
                    };

                    // Look ahead for activation
                    ActivationLayer? actLayer = null;

                    if (i + 1 < graph.Node.Count && IsActivation(graph.Node[i + 1].OpType))
                    {
                        int actInputSize = outputSize;
                        int actOutputSize = outputSize;
                        var actNode = graph.Node[i + 1];
                        actLayer = new ActivationLayer(actInputSize, actOutputSize, MapActivation(actNode.OpType));
                        i += 1; // skip activation node
                    }

                    // Build composite block
                    if (actLayer != null)
                    {
                        var block = new CompositeLayer(dense, actLayer);
                        net.AddLayer(block);
                    }
                    else
                    {
                        net.AddLayer(dense);
                    }
                    i++;
                }
                else if (IsActivation(node.OpType))
                {
                    // Activation without preceding Gemm
                    int actInputSize = 1;
                    int actOutputSize = 1;

                    if (node.Input.Count > 0 && initMap.TryGetValue(node.Input[0], out var inputTensor) && inputTensor.Dims != null && inputTensor.Dims.Count > 0)
                    {
                        actInputSize = (int)inputTensor.Dims[inputTensor.Dims.Count - 1];
                        actOutputSize = actInputSize;
                    }
                    else if (node.Output.Count > 0 && initMap.TryGetValue(node.Output[0], out var outTensor) && outTensor.Dims != null && outTensor.Dims.Count > 0)
                    {
                        actOutputSize = (int)outTensor.Dims[outTensor.Dims.Count - 1];
                        actInputSize = actOutputSize;
                    }

                    var act = new ActivationLayer(actInputSize, actOutputSize, MapActivation(node.OpType));
                    net.AddLayer(act);
                    i++;
                }
                else
                {
                    // Skip nodes that aren't layers (Reshape, Transpose, etc.)
                    i++;
                }
            }

            return net;
        }

        /// <summary>
        /// Attempts to parse a MultiHeadAttentionLayer from a sequence of ONNX nodes.
        /// MHA pattern: Gemm(Q) -> Gemm(K) -> Gemm(V) -> Transpose -> MatMul -> Softmax -> MatMul -> Gemm(O)
        /// </summary>
        private bool TryParseMultiHeadAttention(GraphProto graph, int startIdx, Dictionary<string, TensorProto> initMap, 
            out MultiHeadAttentionLayer? mhaLayer, out int nodesConsumed)
        {
            mhaLayer = null;
            nodesConsumed = 0;

            if (startIdx + 7 >= graph.Node.Count)
                return false;

            // Expected pattern:
            // [0] Gemm (Q projection)
            // [1] Gemm (K projection)
            // [2] Gemm (V projection)
            // [3] Transpose (K)
            // [4] MatMul (Q @ K^T)
            // [5] Softmax
            // [6] MatMul (attention @ V)
            // [7] Gemm (O projection)

            var node0 = graph.Node[startIdx];
            var node1 = graph.Node[startIdx + 1];
            var node2 = graph.Node[startIdx + 2];
            var node3 = graph.Node[startIdx + 3];
            var node4 = graph.Node[startIdx + 4];
            var node5 = graph.Node[startIdx + 5];
            var node6 = graph.Node[startIdx + 6];
            var node7 = graph.Node[startIdx + 7];

            // Verify pattern
            if (node0.OpType != "Gemm" || node1.OpType != "Gemm" || node2.OpType != "Gemm" ||
                node3.OpType != "Transpose" || node4.OpType != "MatMul" || node5.OpType != "Softmax" ||
                node6.OpType != "MatMul" || node7.OpType != "Gemm")
            {
                return false;
            }

            // Extract weight dimensions from the first Gemm (Q projection)
            if (node0.Input.Count < 3 || !initMap.TryGetValue(node0.Input[1], out var wQTensor) ||
                !initMap.TryGetValue(node0.Input[2], out var bQTensor))
            {
                return false;
            }

            // Extract bias to get modelDim
            if (bQTensor.FloatData.Count == 0)
                return false;

            int modelDim = bQTensor.FloatData.Count;
            
            // Extract Q, K, V, O weights and biases
            if (!ExtractMHAWeights(initMap, node0, node1, node2, node7, modelDim, 
                out double[] weightsQ, out double[] weightsK, out double[] weightsV, out double[] weightsO,
                out double[] biasQ, out double[] biasK, out double[] biasV, out double[] biasO))
            {
                return false;
            }

            // Create the MultiHeadAttentionLayer
            try
            {
                var mha = new MultiHeadAttentionLayer(modelDim, Math.Max(1, modelDim / 64)); // Estimate headCount
                
                // Combine weights and biases in the expected format
                double[] combinedWeights = new double[weightsQ.Length + weightsK.Length + weightsV.Length + weightsO.Length];
                double[] combinedBiases = new double[biasQ.Length + biasK.Length + biasV.Length + biasO.Length];

                Array.Copy(weightsQ, 0, combinedWeights, 0, weightsQ.Length);
                Array.Copy(weightsK, 0, combinedWeights, weightsQ.Length, weightsK.Length);
                Array.Copy(weightsV, 0, combinedWeights, weightsQ.Length + weightsK.Length, weightsV.Length);
                Array.Copy(weightsO, 0, combinedWeights, weightsQ.Length + weightsK.Length + weightsV.Length, weightsO.Length);

                Array.Copy(biasQ, 0, combinedBiases, 0, biasQ.Length);
                Array.Copy(biasK, 0, combinedBiases, biasQ.Length, biasK.Length);
                Array.Copy(biasV, 0, combinedBiases, biasQ.Length + biasK.Length, biasV.Length);
                Array.Copy(biasO, 0, combinedBiases, biasQ.Length + biasK.Length + biasV.Length, biasO.Length);

                mha.Weights = combinedWeights;
                mha.Biases = combinedBiases;

                mhaLayer = mha;
                nodesConsumed = 8;
                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Extracts weight and bias arrays from MHA Gemm nodes
        /// </summary>
        private bool ExtractMHAWeights(Dictionary<string, TensorProto> initMap, NodeProto qNode, NodeProto kNode, 
            NodeProto vNode, NodeProto oNode, int modelDim,
            out double[] weightsQ, out double[] weightsK, out double[] weightsV, out double[] weightsO,
            out double[] biasQ, out double[] biasK, out double[] biasV, out double[] biasO)
        {
            weightsQ = weightsK = weightsV = weightsO = Array.Empty<double>();
            biasQ = biasK = biasV = biasO = Array.Empty<double>();

            try
            {
                // Extract Q weights/bias
                if (qNode.Input.Count < 3 || !initMap.TryGetValue(qNode.Input[1], out var wQ) || 
                    !initMap.TryGetValue(qNode.Input[2], out var bQ))
                    return false;

                weightsQ = [.. wQ.FloatData.Select(f => (double)f)];
                biasQ = [.. bQ.FloatData.Select(f => (double)f)];

                // Extract K weights/bias
                if (kNode.Input.Count < 3 || !initMap.TryGetValue(kNode.Input[1], out var wK) || 
                    !initMap.TryGetValue(kNode.Input[2], out var bK))
                    return false;

                weightsK = [.. wK.FloatData.Select(f => (double)f)];
                biasK = [.. bK.FloatData.Select(f => (double)f)];

                // Extract V weights/bias
                if (vNode.Input.Count < 3 || !initMap.TryGetValue(vNode.Input[1], out var wV) || 
                    !initMap.TryGetValue(vNode.Input[2], out var bV))
                    return false;

                weightsV = [.. wV.FloatData.Select(f => (double)f)];
                biasV = [.. bV.FloatData.Select(f => (double)f)];

                // Extract O weights/bias
                if (oNode.Input.Count < 3 || !initMap.TryGetValue(oNode.Input[1], out var wO) || 
                    !initMap.TryGetValue(oNode.Input[2], out var bO))
                    return false;

                weightsO = [.. wO.FloatData.Select(f => (double)f)];
                biasO = [.. bO.FloatData.Select(f => (double)f)];

                return true;
            }
            catch
            {
                return false;
            }
        }

        #region Support Functions
        private static void GetNodeAttributes(Dictionary<string, TensorProto> initMap, NodeProto node, out double[] W, out double[] B, out int outputSize, out int inputSize)
        {
            // Validate inputs exist on node
            if (node.Input == null || node.Input.Count < 3)
                throw new ArgumentException($"Node '{node.OpType}' does not have the expected weight and bias inputs.");

            string wName = node.Input[1];
            string bName = node.Input[2];

            if (!initMap.TryGetValue(wName, out var wTensor))
                throw new KeyNotFoundException($"Weight initializer '{wName}' not found for node '{node.OpType}'.");

            if (!initMap.TryGetValue(bName, out var bTensor))
                throw new KeyNotFoundException($"Bias initializer '{bName}' not found for node '{node.OpType}'.");

            // Expect FloatData to be populated (tests create FloatData). If missing, provide clearer error.
            if (wTensor.FloatData == null || wTensor.FloatData.Count == 0)
                throw new InvalidOperationException($"Weight initializer '{wName}' contains no FloatData.");
            if (bTensor.FloatData == null || bTensor.FloatData.Count == 0)
                throw new InvalidOperationException($"Bias initializer '{bName}' contains no FloatData.");

            W = [.. wTensor.FloatData.Select(f => (double)f)];
            B = [.. bTensor.FloatData.Select(f => (double)f)];
            outputSize = B.Length;
            inputSize = W.Length / outputSize;
        }

        private string ExportCompositeLayer(GraphProto graph, string prevOutput, CompositeLayer layer, ref int globalNodeIndex)
        {
            foreach (var subLayer in layer.SubLayers)
            {
                if (subLayer is DenseLayer dense)
                {
                    string wName = $"W{globalNodeIndex}";
                    string bName = $"B{globalNodeIndex}";
                    string gemmOut = $"GemmOut{globalNodeIndex}";

                    graph.Initializer.Add(MakeTensor(wName, dense.Weights, [(long)dense.Biases.Length, dense.Weights.Length / dense.Biases.Length]));
                    graph.Initializer.Add(MakeTensor(bName, dense.Biases, [(long)dense.Biases.Length]));

                    graph.Node.Add(new NodeProto
                    {
                        OpType = "Gemm",
                        Input = { prevOutput, wName, bName },
                        Output = { gemmOut }
                    });

                    prevOutput = gemmOut;
                    globalNodeIndex++;
                }
                else if (subLayer is ActivationLayer act)
                {
                    string actOut = $"ActOut{globalNodeIndex}";

                    graph.Node.Add(new NodeProto
                    {
                        OpType = MapActivation(act.ActivationFunction),
                        Input = { prevOutput },
                        Output = { actOut }
                    });

                    prevOutput = actOut;
                    globalNodeIndex++;
                }
                else if (subLayer is LayerNormLayer)
                {
                    // LayerNormLayer has no trainable parameters, export as InstanceNormalization
                    string normOut = $"NormOut{globalNodeIndex}";
                    graph.Node.Add(new NodeProto
                    {
                        OpType = "InstanceNormalization",
                        Input = { prevOutput },
                        Output = { normOut },
                        Attribute =
                        {
                            new AttributeProto { Name = "epsilon", F = 1e-5f }
                        }
                    });
                    prevOutput = normOut;
                    globalNodeIndex++;
                }
                else if (subLayer is MultiHeadAttentionLayer mha)
                {
                    // Export MultiHeadAttentionLayer weights as Gemm operations
                    prevOutput = ExportMultiHeadAttention(graph, prevOutput, mha, ref globalNodeIndex);
                }
                else if (subLayer is FeedForwardLayer ff)
                {
                    // Recursively export FeedForwardLayer as a CompositeLayer
                    prevOutput = ExportCompositeLayer(graph, prevOutput, ff, ref globalNodeIndex);
                }
                else if (subLayer is ResidualAddLayer residual)
                {
                    // ResidualAddLayer is a skip connection - in ONNX we use Add node
                    // Note: This is simplified and assumes residual connections are handled at a higher level
                    // For full support, the graph structure would need to track branch paths
                }
                else
                {
                    // Other composite or unknown layer types
                    if (subLayer is CompositeLayer composite)
                    {
                        prevOutput = ExportCompositeLayer(graph, prevOutput, composite, ref globalNodeIndex);
                    }
                }
            }
            return prevOutput;
        }

        private string ExportMultiHeadAttention(GraphProto graph, string prevOutput, MultiHeadAttentionLayer mha, ref int globalNodeIndex)
        {
            // MultiHeadAttentionLayer has 4 weight matrices (Q, K, V, O projections) and 4 bias vectors
            // Each projection is modelDim x modelDim
            // For ONNX export, we decompose the attention into linear projections + attention computation
            
            int modelDim = (int)mha.InputSize;
            int block = modelDim * modelDim;
            
            // Extract the four projection weight matrices from the layer's weight array
            double[] weightsQ = mha.Weights.AsSpan(0, block).ToArray();
            double[] weightsK = mha.Weights.AsSpan(block, block).ToArray();
            double[] weightsV = mha.Weights.AsSpan(block * 2, block).ToArray();
            double[] weightsO = mha.Weights.AsSpan(block * 3, block).ToArray();
            
            double[] biasQ = mha.Biases.AsSpan(0, modelDim).ToArray();
            double[] biasK = mha.Biases.AsSpan(modelDim, modelDim).ToArray();
            double[] biasV = mha.Biases.AsSpan(modelDim * 2, modelDim).ToArray();
            double[] biasO = mha.Biases.AsSpan(modelDim * 3, modelDim).ToArray();
            
            // Create Gemm nodes for Q, K, V projections
            string wQName = $"W_Q{globalNodeIndex}";
            string bQName = $"B_Q{globalNodeIndex}";
            string qOut = $"Q_Out{globalNodeIndex}";
            
            graph.Initializer.Add(MakeTensor(wQName, weightsQ, [(long)modelDim, (long)modelDim]));
            graph.Initializer.Add(MakeTensor(bQName, biasQ, [(long)modelDim]));
            graph.Node.Add(new NodeProto
            {
                OpType = "Gemm",
                Input = { prevOutput, wQName, bQName },
                Output = { qOut }
            });
            
            string wKName = $"W_K{globalNodeIndex}";
            string bKName = $"B_K{globalNodeIndex}";
            string kOut = $"K_Out{globalNodeIndex}";
            
            graph.Initializer.Add(MakeTensor(wKName, weightsK, [(long)modelDim, (long)modelDim]));
            graph.Initializer.Add(MakeTensor(bKName, biasK, [(long)modelDim]));
            graph.Node.Add(new NodeProto
            {
                OpType = "Gemm",
                Input = { prevOutput, wKName, bKName },
                Output = { kOut }
            });
            
            string wVName = $"W_V{globalNodeIndex}";
            string bVName = $"B_V{globalNodeIndex}";
            string vOut = $"V_Out{globalNodeIndex}";
            
            graph.Initializer.Add(MakeTensor(wVName, weightsV, [(long)modelDim, (long)modelDim]));
            graph.Initializer.Add(MakeTensor(bVName, biasV, [(long)modelDim]));
            graph.Node.Add(new NodeProto
            {
                OpType = "Gemm",
                Input = { prevOutput, wVName, bVName },
                Output = { vOut }
            });
            
            // Create MatMul for scaled dot-product (simplified: just use MatMul K^T and Q)
            string kTransposed = $"K_T{globalNodeIndex}";
            graph.Node.Add(new NodeProto
            {
                OpType = "Transpose",
                Input = { kOut },
                Output = { kTransposed }
            });
            
            string scores = $"Scores{globalNodeIndex}";
            graph.Node.Add(new NodeProto
            {
                OpType = "MatMul",
                Input = { qOut, kTransposed },
                Output = { scores }
            });
            
            // Softmax over scores
            string attention = $"Attention{globalNodeIndex}";
            graph.Node.Add(new NodeProto
            {
                OpType = "Softmax",
                Input = { scores },
                Output = { attention },
                Attribute = { new AttributeProto { Name = "axis", I = 1 } }
            });
            
            // MatMul attention with V
            string attnV = $"AttnV{globalNodeIndex}";
            graph.Node.Add(new NodeProto
            {
                OpType = "MatMul",
                Input = { attention, vOut },
                Output = { attnV }
            });
            
            // Output projection
            string wOName = $"W_O{globalNodeIndex}";
            string bOName = $"B_O{globalNodeIndex}";
            string mhaOut = $"MHA_Out{globalNodeIndex}";
            
            graph.Initializer.Add(MakeTensor(wOName, weightsO, [(long)modelDim, (long)modelDim]));
            graph.Initializer.Add(MakeTensor(bOName, biasO, [(long)modelDim]));
            graph.Node.Add(new NodeProto
            {
                OpType = "Gemm",
                Input = { attnV, wOName, bOName },
                Output = { mhaOut }
            });
            
            globalNodeIndex++;
            return mhaOut;
        }

        private string ExportBasicLayer(GraphProto graph, string prevOutput, ref int globalNodeIndex, ILayer layer)
        {
            string wName = $"W{globalNodeIndex}";
            string bName = $"B{globalNodeIndex}";
            string gemmOut = $"GemmOut{globalNodeIndex}";
            string actOut = $"ActOut{globalNodeIndex}";

            // Add weights
            graph.Initializer.Add(MakeTensor(wName, layer.Weights, [.. new[] { layer.OutputSize, layer.InputSize }.Select(x => (long)x)]));
            graph.Initializer.Add(MakeTensor(bName, layer.Biases, [.. new[] { layer.OutputSize }.Select(x => (long)x)]));

            // Gemm node
            graph.Node.Add(new NodeProto
            {
                OpType = "Gemm",
                Input = { prevOutput, wName, bName },
                Output = { gemmOut }
            });

            // Activation node
            graph.Node.Add(new NodeProto
            {
                OpType = MapActivation(layer.ActivationFunction),
                Input = { gemmOut },
                Output = { actOut }
            });

            globalNodeIndex++;
            prevOutput = actOut;
            return prevOutput;
        }


        private TensorProto MakeTensor(string name, double[] data, long[] dims)
        {
            var t = new TensorProto
            {
                Name = name,
                DataType = (int)TensorProto.Types.DataType.Float,
            };
            t.Dims.AddRange(dims);
            t.FloatData.AddRange(Array.ConvertAll(data, d => (float)d));
            return t;
        }

        private TypeProto MakeTensorTypeProto(TensorProto.Types.DataType type, long[] dims)
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

        private string MapActivation(ActivationFunction f)
        {
            return f switch
            {
                ActivationFunction.ReLU => "Relu",
                ActivationFunction.LeakyReLU => "LeakyRelu",
                ActivationFunction.Sigmoid => "Sigmoid",
                ActivationFunction.Tanh => "Tanh",
                ActivationFunction.Swish => "Swish",
                ActivationFunction.HardSwish => "HardSwish",
                ActivationFunction.ELU => "ELU",
                ActivationFunction.GELU => "GELU",
                ActivationFunction.Softmax => "Softmax",
                _ => "Identity"
            };
        }
        private ActivationFunction MapActivation(string op)
        {
            return op switch
            {
                "Relu" => ActivationFunction.ReLU,
                "LeakyRelu" => ActivationFunction.LeakyReLU,
                "Sigmoid" => ActivationFunction.Sigmoid,
                "Tanh" => ActivationFunction.Tanh,
                "Swish" => ActivationFunction.Swish,
                "HardSwish" => ActivationFunction.HardSwish,
                "ELU" => ActivationFunction.ELU,
                "GELU" => ActivationFunction.GELU,
                "Softmax" => ActivationFunction.Softmax,
                _ => ActivationFunction.Linear
            };
        }
        private bool IsActivation(string op) =>
        op is "Relu" or "LeakyRelu" or "Tanh" or "Sigmoid" or "Softmax" or "Swish" or "HardSwish" or "ELU" or "GELU";
        #endregion
    }
}
