using Google.Protobuf;
using Onnx;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
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

            var graph = new GraphProto { Name = "HomebrewGraph" };

            string prevOutput = "input";

            // Define model input (you can adjust shape)
            graph.Input.Add(new ValueInfoProto
            {
                Name = "input",
                Type = MakeTensorTypeProto(Onnx.TensorProto.Types.DataType.Float, new[] { 1, network.InputSize })
            });

            for (int i = 0; i < network.Layers.Count; i++)
            {
                LayerBase layer = (LayerBase)network.Layers[i];
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

            // Build lookup for initializers
            var initMap = graph.Initializer.ToDictionary(i => i.Name);

            // Iterate nodes in pairs: Gemm → Activation
            for (int i = 0; i < graph.Node.Count; i++)
            {
                var node = graph.Node[i];
                if (node.OpType != "Gemm") continue;

                string wName = node.Input[1];
                string bName = node.Input[2];

                var W = initMap[wName].FloatData.Select(f => (double)f).ToArray();
                var B = initMap[bName].FloatData.Select(f => (double)f).ToArray();

                // Next node should be activation
                var actNode = graph.Node[i + 1];
                var act = MapActivation(actNode.OpType);

                var layer = new ActivationLayer(W, B, act);
                net.AddLayer(layer);
            }

            return net;
        }

        #region Support Functions
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
        #endregion
    }
}
