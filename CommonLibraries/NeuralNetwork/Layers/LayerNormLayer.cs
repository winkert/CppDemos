using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TRW.CommonLibraries.NeuralNetwork
{
    public class LayerNormLayer : LayerBase
    {
        private readonly int dim;
        private double[] lastInput;

        public LayerNormLayer() : base() { }
        public LayerNormLayer(int dim)
            : base(dim, dim, ActivationFunction.Linear)
        {
            this.dim = dim;
            // No weights or biases for basic LayerNorm
            Weights = [];
            Biases = [];
        }

        public override double[] Forward(double[] input)
        {
            lastInput = input;

            double mean = input.Average();
            double variance = input.Select(v => (v - mean) * (v - mean)).Average();
            double std = Math.Sqrt(variance + 1e-5);

            double[] output = new double[dim];
            for (int i = 0; i < dim; i++)
                output[i] = (input[i] - mean) / std;

            return output;
        }

        public override double[] Backward(double[] input, double[] upstream, double[] dW, double[] db, double lr)
            => upstream;

        public override void UpdateWeights(double[] dW, double[] db, double lr, double l2)
        {
            // No-op
        }
    }


}
