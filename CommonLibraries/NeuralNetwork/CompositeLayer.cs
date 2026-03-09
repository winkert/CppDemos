using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TRW.CommonLibraries.NeuralNetwork
{
    public class CompositeLayer : ActivationLayer
    {
        // Cache forward activations
        private double[][] _forwardCache;
        public List<LayerBase> SubLayers { get; private set; }
        public CompositeLayer(params LayerBase[] subLayers)
            : this(ActivationFunction.ReLU, subLayers)
        {        }
        public CompositeLayer(ActivationFunction function, params LayerBase[] subLayers)
            : base(subLayers[0].InputSize, subLayers.Last().OutputSize, function)
        {
            SubLayers = [.. subLayers];
        }
        public override double[] Forward(double[] inputs)
        {
            _forwardCache = new double[SubLayers.Count + 1][];
            _forwardCache[0] = inputs; // cache input for backward pass
            double[] output = inputs;
            for (int i = 0; i < SubLayers.Count; i++)
            {
                output = SubLayers[i].Forward(output);
                _forwardCache[i + 1] = output;
            }
            return output;
        }
        public override double[] Backward(double[] input, double[] upstream, double[] dW, double[] db, double learningRate)
        {
            // upstream is the gradient from the next layer
            double[] grad = upstream;

            // Walk backwards through sublayers
            for (int i = SubLayers.Count - 1; i >= 0; i--)
            {
                var layer = SubLayers[i];

                double[] layerInput = _forwardCache[i];

                // Each sublayer updates its own weights/biases internally
                grad = layer.Backward(layerInput, grad, dW, db, learningRate);
            }

            return grad; // gradient wrt composite input
        }
        public override void UpdateWeights(double[] dW, double[] db, double learningRate, double l2)
        {
            throw new NotImplementedException("UpdateWeights for CompositeLayer is not implemented. You may need to implement it by calling UpdateWeights on each sub-layer.");
        }
    }
}
