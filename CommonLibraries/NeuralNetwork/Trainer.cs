using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace TRW.CommonLibraries.NeuralNetwork
{
    public class Trainer
    {
        NeuralNetwork _neuralNetwork;
        private double[] _inputMin;
        private double[] _inputMax;
        private double[] _outputMin;
        private double[] _outputMax;

        private Dictionary<int, Dictionary<string, int>> _categoricalMaps = [];
        private Dictionary<string, int> _labelToIndex = [];
        private Dictionary<int, string> _indexToLabel = [];

        public Trainer()
        {
            _neuralNetwork = new NeuralNetwork();
        }

        public void AddLayer(int neuronCount, ActivationFunction activation)
        {
            AddLayer(neuronCount, neuronCount, activation);
        }

        public void AddLayer(int neuronCount, int outputCount, ActivationFunction activation)
        {
            _neuralNetwork.AddLayer(neuronCount, outputCount, activation);
        }

        public void TrainCategorical(List<string[]> categoricalInputs, List<double[]> rawOutputs, int epochs = 1000, double learningRate = 0.01)
        {
            List<double[]> encodedInputs = EncodeCategoricalInputs(categoricalInputs);
            Train(encodedInputs, rawOutputs, epochs, learningRate);
        }

        public void TrainWithOneHotOutput(List<string[]> categoricalInputs, List<string> classLabels, int epochs = 1000, double learningRate = 0.01)
        {
            List<double[]> encodedInputs = EncodeCategoricalInputs(categoricalInputs);
            List<double[]> encodedOutputs = EncodeOneHotOutputs(classLabels);
            Train(encodedInputs, encodedOutputs, epochs, learningRate);
        }

        public void Train(List<double[]> rawInputs, List<double[]> rawOutputs, int epochs = 1000, double learningRate = 0.01)
        {
            if (rawInputs.Count != rawOutputs.Count)
                throw new ArgumentException("Input and output counts must match.");

            NormalizeData(rawInputs, rawOutputs, out List<double[]>? normInputs, out List<double[]>? normOutputs);
            _neuralNetwork.TrainBatch(normInputs, normOutputs, learningRate, 1e-4, epochs, true);
        }

        private void NormalizeData(List<double[]> inputs, List<double[]> outputs, out List<double[]> normInputs, out List<double[]> normOutputs)
        {
            int inputDim = inputs[0].Length;
            int outputDim = outputs[0].Length;

            _inputMin = new double[inputDim];
            _inputMax = new double[inputDim];
            _outputMin = new double[outputDim];
            _outputMax = new double[outputDim];

            for (int i = 0; i < inputDim; i++)
            {
                _inputMin[i] = inputs.Min(x => x[i]);
                _inputMax[i] = inputs.Max(x => x[i]);
            }

            for (int i = 0; i < outputDim; i++)
            {
                _outputMin[i] = outputs.Min(x => x[i]);
                _outputMax[i] = outputs.Max(x => x[i]);
            }

            normInputs = [.. inputs.Select(x => NormalizeVector(x, _inputMin, _inputMax))];
            normOutputs = [.. outputs.Select(y => NormalizeVector(y, _outputMin, _outputMax))];
        }

        private double[] NormalizeVector(double[] vector, double[] min, double[] max)
        {
            return [.. vector.Select((val, i) =>
                (max[i] - min[i]) == 0 ? 0.0 : (val - min[i]) / (max[i] - min[i])
            )];
        }

        private double[] DenormalizeVector(double[] vector, double[] min, double[] max)
        {
            return [.. vector.Select((val, i) =>
                val * (max[i] - min[i]) + min[i]
            )];
        }

        private List<double[]> EncodeCategoricalInputs(List<string[]> inputs)
        {
            int featureCount = inputs[0].Length;
            _categoricalMaps.Clear();

            // Build encoding maps
            for (int i = 0; i < featureCount; i++)
            {
                List<string> unique = [.. inputs.Select(x => x[i]).Distinct()];
                Dictionary<string, int> map = [];
                for (int j = 0; j < unique.Count; j++)
                    map[unique[j]] = j;
                _categoricalMaps[i] = map;
            }

            // One-hot encode
            List<double[]> encoded = [];
            foreach (string[] row in inputs)
            {
                List<double> vector = [];
                for (int i = 0; i < row.Length; i++)
                {
                    int categoryCount = _categoricalMaps[i].Count;
                    double[] oneHot = new double[categoryCount];
                    oneHot[_categoricalMaps[i][row[i]]] = 1.0;
                    vector.AddRange(oneHot);
                }
                encoded.Add([.. vector]);
            }

            return encoded;
        }

        private List<double[]> EncodeOneHotOutputs(List<string> labels)
        {
            List<string> uniqueLabels = [.. labels.Distinct()];
            _labelToIndex = uniqueLabels.Select((label, idx) => new { label, idx })
                                        .ToDictionary(x => x.label, x => x.idx);
            _indexToLabel = _labelToIndex.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

            int classCount = uniqueLabels.Count;
            List<double[]> encoded = [];

            foreach (string label in labels)
            {
                double[] oneHot = new double[classCount];
                oneHot[_labelToIndex[label]] = 1.0;
                encoded.Add(oneHot);
            }

            return encoded;
        }

        public string PredictClass(string[] input)
        {
            List<double> encoded = [];
            for (int i = 0; i < input.Length; i++)
            {
                int categoryCount = _categoricalMaps[i].Count;
                double[] oneHot = new double[categoryCount];
                oneHot[_categoricalMaps[i][input[i]]] = 1.0;
                encoded.AddRange(oneHot);
            }

            double[] output = _neuralNetwork.Forward([.. encoded]);
            int predictedIndex = Array.IndexOf(output, output.Max());
            return _indexToLabel[predictedIndex];
        }

        public double[] Predict(double[] input)
        {
            double[] normInput = NormalizeVector(input, _inputMin, _inputMax);
            double[] normOutput = _neuralNetwork.Forward(normInput);
            return DenormalizeVector(normOutput, _outputMin, _outputMax);
        }

    }
}
