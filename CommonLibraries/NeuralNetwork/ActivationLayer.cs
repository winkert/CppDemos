namespace TRW.CommonLibraries.NeuralNetwork
{
    public class ActivationLayer : LayerBase
    {
        public ActivationLayer(long inputSize, long outputSize, ActivationFunction activationFunction) 
            : base(inputSize, outputSize, activationFunction)
        {
        }

        public ActivationLayer(double[] weights, double[] biases, ActivationFunction activationFunction)
            : base(weights.Length / biases.Length, biases.Length, activationFunction)
        {
            if (weights.Length != biases.Length * InputSize)
            {
                throw new ArgumentException("Weights length must be equal to biases length multiplied by input size.");
            }
            Weights = weights;
            Biases = biases;
            ActivationFunction = activationFunction;
        }
        public override double[] Forward(double[] inputs)
        {
            if(inputs.Length != InputSize)
            {
                throw new ArgumentException("Input size does not match layer input size.");
            }

            double[] z = PreActivationVector;
            for (int o = 0; o < OutputSize; ++o)
            {
                double s = 0.0;
                long baseIdx = o * InputSize;
                for (int i = 0; i < InputSize; ++i)
                {
                    s += Weights[baseIdx + i] * inputs[i];
                }
                z[o] = s + Biases[o];
            }

            // compute activations and copy into PostActivationVector so Backward sees them
            double[] activated = ActivateArray(z, ActivationFunction);
            for (int i = 0; i < activated.Length; ++i)
                PostActivationVector[i] = activated[i];

            return PostActivationVector;
        }

        public override double[] Backward(double[] input, double[] upstream, double[] dW, double[] db, double learningRate)
        {
            if (input.Length != InputSize)
            {
                throw new ArgumentException("Input size does not match layer input size.");
            }

            double[] z = PreActivationVector;
            double[] a = PostActivationVector;
            double[] dZ = new double[OutputSize];
            double[] deltaPrev = new double[InputSize];
            if(ActivationFunction == ActivationFunction.Softmax)
            {
                // upstream already yPred - yTrue (for softmax + cross-entropy)
                dZ = upstream;
            }
            else
            {
                for(int o = 0; o < OutputSize; ++o)
                {
                    dZ[o] = upstream[o] * ActivateDerivative(z[o], a[o], ActivationFunction);
                }
            }

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
            for(int o = 0; o < OutputSize; ++o)
            {
                long baseIdx = o * (InputSize);
                for(int i = 0; i < InputSize; ++i)
                {
                    Weights[baseIdx + i] -= learningRate * (dW[baseIdx + i] + l2 * Weights[baseIdx + i]);
                }
                Biases[o] -= learningRate * db[o];
            }
        }
    }
}
