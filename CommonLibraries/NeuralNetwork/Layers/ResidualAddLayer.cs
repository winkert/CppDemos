using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TRW.CommonLibraries.NeuralNetwork
{
    public class ResidualAddLayer : LayerBase
    {
        public ResidualAddLayer(int size)
            : base(size, size, ActivationFunction.Linear)
        {
            Weights = [];
            Biases = [];
        }

        public double[] Forward(double[] input, double[] sublayerOutput)
        {
            double[] result = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
                result[i] = input[i] + sublayerOutput[i];
            return result;
        }

        public override double[] Forward(double[] input)
        {
            // Identity if used standalone
            return input;
        }

        public override double[] Backward(double[] input, double[] upstream, double[] dW, double[] db, double lr)
            => upstream;

        public override void UpdateWeights(double[] dW, double[] db, double lr, double l2)
        {
            // No-op
        }
    }

}
