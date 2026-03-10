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
                Type = MakeTensorTypeProto(Onnx.TensorProto.Types.DataType.Float, new[] { 1, network.InputSize })
            });

            for (int i = 0; i < network.Layers.Count; i++)
            {
                ILayer layer = (LayerBase)network.Layers[i];
                if (layer is CompositeLayer)
                {
                    prevOutput = ExportCompositeLayer(graph, prevOutput, (CompositeLayer)layer);
                }
                else
                {
                    prevOutput = ExportBasicLayer(graph, prevOutput, i, layer);
                }
            }

            // Define model output
            graph.Output.Add(new ValueInfoProto
            {
                Name = prevOutput,
                Type = MakeTensorTypeProto(Onnx.TensorProto.Types.DataType.Float, new[] { 1, network.OutputSize })
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
                // Case 1: Dense layer (Gemm)
                if (node.OpType == "Gemm")
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
                        var actNode = graph.Node[i + 1];
                        GetNodeAttributes(initMap, actNode, out _, out _, out int actOutputSize, out int actInputSize);

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
                        // No activation, just add dense
                        net.AddLayer(dense);
                    }
                }
                else if (IsActivation(node.OpType))
                {
                    GetNodeAttributes(initMap, node, out W, out B, out outputSize, out inputSize);

                    // Activation without preceding Gemm — rare but possible
                    var act = new ActivationLayer(inputSize, outputSize, MapActivation(node.OpType));
                    net.AddLayer(act);
                }
                else
                {
                    // Skip nodes that aren't layers (Reshape, Transpose, etc.)
                }

                i++;
            }

            return net;
        }

        #region Support Functions
        private static void GetNodeAttributes(Dictionary<string, TensorProto> initMap, NodeProto node, out double[] W, out double[] B, out int outputSize, out int inputSize)
        {
            string wName = node.Input[1];
            string bName = node.Input[2];

            W = initMap[wName].FloatData.Select(f => (double)f).ToArray();
            B = initMap[bName].FloatData.Select(f => (double)f).ToArray();
            outputSize = B.Length;
            inputSize = W.Length / outputSize;
        }

        private string ExportCompositeLayer(GraphProto graph, string prevOutput, CompositeLayer layer)
        {
            int nodeIndex = 0;
            foreach (var subLayer in layer.SubLayers)
            {
                if (subLayer is DenseLayer dense)
                {
                    string wName = $"W{nodeIndex}";
                    string bName = $"B{nodeIndex}";
                    string gemmOut = $"GemmOut{nodeIndex}";

                    graph.Initializer.Add(MakeTensor(wName, dense.Weights, new[] { (long)dense.Biases.Length, dense.Weights.Length / dense.Biases.Length }));
                    graph.Initializer.Add(MakeTensor(bName, dense.Biases, new[] { (long)dense.Biases.Length }));

                    graph.Node.Add(new NodeProto
                    {
                        OpType = "Gemm",
                        Input = { prevOutput, wName, bName },
                        Output = { gemmOut }
                    });

                    prevOutput = gemmOut;
                    nodeIndex++;
                }
                else if (subLayer is ActivationLayer act)
                {
                    string actOut = $"ActOut{nodeIndex}";

                    graph.Node.Add(new NodeProto
                    {
                        OpType = MapActivation(act.ActivationFunction),
                        Input = { prevOutput },
                        Output = { actOut }
                    });

                    prevOutput = actOut;
                    nodeIndex++;
                }
            }
            return prevOutput;
        }

        private string ExportBasicLayer(GraphProto graph, string prevOutput, int i, ILayer layer)
        {
            string wName = $"W{i}";
            string bName = $"B{i}";
            string gemmOut = $"GemmOut{i}";
            string actOut = $"ActOut{i}";

            // Add weights
            graph.Initializer.Add(MakeTensor(wName, layer.Weights, [layer.OutputSize, layer.InputSize]));
            graph.Initializer.Add(MakeTensor(bName, layer.Biases, [layer.OutputSize]));

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
                ElemType = (int)type
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
        op is "Relu" or "Tanh" or "Sigmoid" or "Softmax" or "Gelu";
        #endregion
    }
}
