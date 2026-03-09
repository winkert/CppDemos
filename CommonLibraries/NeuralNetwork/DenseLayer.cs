using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TRW.CommonLibraries.NeuralNetwork
{
    public class DenseLayer : LayerBase
    {
        private readonly int _inputSize;
        private readonly int _outputSize;

        public DenseLayer(int inputSize, int outputSize)
            : base(inputSize, outputSize, ActivationFunction.Linear)
        {
            _inputSize = inputSize;
            _outputSize = outputSize;
            Weights = new double[inputSize * outputSize];
            Biases = new double[outputSize];
        }

        public override double[] Forward(double[] input)
        {
            double[] output = new double[_outputSize];

            for (int o = 0; o < _outputSize; o++)
            {
                double sum = Biases[o];
                for (int i = 0; i < _inputSize; i++)
                    sum += Weights[o * _inputSize + i] * input[i];

                output[o] = sum;
            }

            return output;
        }

        public override double[] Backward(double[] input,
                                 double[] upstream,
                                 double[] dW,
                                 double[] db,
                                 double learningRate)
        {
            double[] gradInput = new double[_inputSize];

            // Compute gradients
            for (int o = 0; o < _outputSize; o++)
            {
                db[o] += upstream[o];

                for (int i = 0; i < _inputSize; i++)
                {
                    dW[o * _inputSize + i] += upstream[o] * input[i];
                    gradInput[i] += upstream[o] * Weights[o * _inputSize + i];
                }
            }

            // SGD update
            for (int i = 0; i < Weights.Length; i++)
                Weights[i] -= learningRate * dW[i];

            for (int i = 0; i < Biases.Length; i++)
                Biases[i] -= learningRate * db[i];

            return gradInput;
        }

        public override void UpdateWeights(double[] dW, double[] db, double learningRate, double l2)
        {
            throw new NotImplementedException();
        }
    }

}
