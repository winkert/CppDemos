using System;

namespace TRW.CommonLibraries.NeuralNetwork
{
    public class DenseLayer : LayerBase
    {
        public DenseLayer() : base()
        {
        }

        public DenseLayer(long inputSize, long outputSize)
            : base(inputSize, outputSize, ActivationFunction.Linear)
        {
        }

        public DenseLayer(long inputSize, long outputSize, ActivationFunction activation)
            : base(inputSize, outputSize, activation) { }

        public DenseLayer(double[] weights, double[] biases)
            : base(weights.Length / biases.Length, biases.Length, ActivationFunction.Linear)
        {
            if (weights.Length != biases.Length * InputSize)
                throw new ArgumentException("Weights length must be equal to biases length multiplied by input size.");
            Weights = weights;
            Biases = biases;
            ActivationFunction = ActivationFunction.Linear;
        }

        public override double[] Forward(double[] inputs)
        {
            if (inputs.Length != InputSize)
                throw new ArgumentException($"Input size [{inputs.Length}] does not match layer input size [{InputSize}].");

            double[] z = PreActivationVector;
            for (int o = 0; o < OutputSize; ++o)
            {
                double s = 0.0;
                long baseIdx = o * InputSize;
                for (int i = 0; i < InputSize; ++i)
                    s += Weights[baseIdx + i] * inputs[i];
                z[o] = s + Biases[o];
            }

            // For a pure dense/linear layer post-activation == pre-activation
            for (int i = 0; i < z.Length; ++i)
                PostActivationVector[i] = z[i];

            return PostActivationVector;
        }

        public override double[] Backward(double[] input, double[] upstream, double[] dW, double[] db, double learningRate)
        {
            if (input.Length != InputSize)
                throw new ArgumentException("Input size does not match layer input size.");

            // Linear layer: dZ == upstream
            double[] dZ = upstream;
            double[] deltaPrev = new double[InputSize];

            for (int o = 0; o < OutputSize; ++o)
            {
                long baseIdx = o * InputSize;
                db[o] += dZ[o];
                for (int i = 0; i < InputSize; ++i)
                {
                    dW[baseIdx + i] += dZ[o] * input[i];
                    deltaPrev[i] += Weights[baseIdx + i] * dZ[o];
                }
            }

            return deltaPrev;
        }

        public override void UpdateWeights(double[] dW, double[] db, double learningRate, double l2)
        {
            for (int o = 0; o < OutputSize; ++o)
            {
                long baseIdx = o * InputSize;
                for (int i = 0; i < InputSize; ++i)
                {
                    Weights[baseIdx + i] -= learningRate * (dW[baseIdx + i] + l2 * Weights[baseIdx + i]);
                }
                Biases[o] -= learningRate * db[o];
            }
        }
    }
}
