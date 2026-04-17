using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TRW.CommonLibraries.NeuralNetwork
{
    public class MultiHeadAttentionLayer : LayerBase
    {
        private readonly int modelDim;
        private readonly int headCount;
        private readonly int headDim;

        private double[] lastQ, lastK, lastV;
        private double[] lastAttention;

        public MultiHeadAttentionLayer(int modelDim, int headCount)
            : base(modelDim, modelDim, ActivationFunction.Linear)
        {
            this.modelDim = modelDim;
            this.headCount = headCount;
            this.headDim = modelDim / headCount;

            // Allocate weights: Q, K, V, O projections
            int block = modelDim * modelDim;
            Weights = new double[block * 4];
            Biases = new double[modelDim * 4];

            Initialize(Weights);
            Initialize(Biases);
        }

        private void Initialize(double[] arr)
        {
            Random r = new Random();
            for (int i = 0; i < arr.Length; i++)
                arr[i] = (r.NextDouble() - 0.5) * 0.02;
        }

        public override double[] Forward(double[] input)
        {
            int block = modelDim * modelDim;

            double[] Wq = Weights.AsSpan(0, block).ToArray();
            double[] Wk = Weights.AsSpan(block, block).ToArray();
            double[] Wv = Weights.AsSpan(block * 2, block).ToArray();
            double[] Wo = Weights.AsSpan(block * 3, block).ToArray();

            double[] bq = Biases.AsSpan(0, modelDim).ToArray();
            double[] bk = Biases.AsSpan(modelDim, modelDim).ToArray();
            double[] bv = Biases.AsSpan(modelDim * 2, modelDim).ToArray();
            double[] bo = Biases.AsSpan(modelDim * 3, modelDim).ToArray();

            lastQ = MatMul(input, Wq, bq);
            lastK = MatMul(input, Wk, bk);
            lastV = MatMul(input, Wv, bv);

            lastAttention = ScaledDotProduct(lastQ, lastK, lastV);

            return MatMul(lastAttention, Wo, bo);
        }

        private double[] MatMul(double[] x, double[] W, double[] b)
        {
            double[] y = new double[b.Length];
            for (int i = 0; i < b.Length; i++)
            {
                double sum = b[i];
                for (int j = 0; j < x.Length; j++)
                    sum += x[j] * W[i * x.Length + j];
                y[i] = sum;
            }
            return y;
        }

        private double[] ScaledDotProduct(double[] Q, double[] K, double[] V)
        {
            double scale = 1.0 / Math.Sqrt(headDim);

            double score = 0;
            for (int i = 0; i < Q.Length; i++)
                score += Q[i] * K[i];

            score *= scale;

            double weight = 1.0 / (1.0 + Math.Exp(-score));

            double[] output = new double[V.Length];
            for (int i = 0; i < V.Length; i++)
                output[i] = weight * V[i];

            return output;
        }

        public override double[] Backward(double[] input, double[] upstream, double[] dW, double[] db, double lr)
        {
            // Minimal backward for now
            return upstream;
        }

        public override void UpdateWeights(double[] dW, double[] db, double lr, double l2)
        {
            for (int i = 0; i < Weights.Length; i++)
                Weights[i] -= lr * (dW[i] + l2 * Weights[i]);

            for (int i = 0; i < Biases.Length; i++)
                Biases[i] -= lr * db[i];
        }
    }



}
