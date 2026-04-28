using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace TRW.CommonLibraries.NeuralNetwork
{
    /// <summary>
    /// CompositeLayer composes an ordered list of ILayer instances and exposes them as a single ILayer.
    /// Weights and biases are presented as the concatenation of the contained layers' weights/biases so
    /// the trainer can allocate a single gradient buffer per composite layer.
    /// </summary>
    public class CompositeLayer : LayerBase
    {
        private double[][] _forwardCache = [];

        public List<ILayer> SubLayers { get; private set; }

        public CompositeLayer()
        {
            SubLayers = new List<ILayer>();
        }

        public CompositeLayer(params ILayer[] subLayers)
        {
            if (subLayers == null || subLayers.Length == 0)
                throw new ArgumentException("CompositeLayer requires at least one sublayer.");

            SubLayers = new List<ILayer>(subLayers);
        }

        public CompositeLayer(int inputSize, int hiddenSize, int outputSize, ActivationFunction activationFunction)
        {
            // By default we'll create a simple 2-layer composite: Dense + Activation
            SubLayers = new List<ILayer>
            {
                new DenseLayer(inputSize, hiddenSize),
                new ActivationLayer(hiddenSize, outputSize, activationFunction)
            };
        }

        public override double[] Weights
        {
            get => [.. SubLayers.SelectMany(l => l.Weights)];
            set
            {
                // in LayerBase constructor we initialize Weights/Biases to empty arrays, so we should allow setting empty arrays here without error even if SubLayers is not yet populated.
                if (SubLayers == null || SubLayers.Count == 0)
                    return;
                int total = SubLayers.Sum(l => l.Weights.Length);
                if (value == null || value.Length != total)
                    throw new ArgumentException("Weights length does not match composite internal sizes.");

                int pos = 0;
                foreach (var l in SubLayers)
                {
                    int len = l.Weights.Length;
                    l.Weights = [.. value.Skip(pos).Take(len)];
                    pos += len;
                }
            }
        }

        public override double[] Biases
        {
            get => [.. SubLayers.SelectMany(l => l.Biases)];
            set
            {
                // in LayerBase constructor we initialize Weights/Biases to empty arrays, so we should allow setting empty arrays here without error even if SubLayers is not yet populated.
                if (SubLayers == null || SubLayers.Count == 0)
                    return;
                int total = SubLayers.Sum(l => l.Biases.Length);
                if (value == null || value.Length != total)
                    throw new ArgumentException("Biases length does not match composite internal sizes.");

                int pos = 0;
                foreach (var l in SubLayers)
                {
                    int len = l.Biases.Length;
                    l.Biases = [.. value.Skip(pos).Take(len)];
                    pos += len;
                }
            }
        }

        public override long InputSize => SubLayers.Count > 0 ? SubLayers.First().InputSize : 0;
        public override long OutputSize => SubLayers.Count > 0 ? SubLayers.Last().OutputSize : 0;
        public override ActivationFunction ActivationFunction => SubLayers.Count > 0 ? SubLayers.Last().ActivationFunction : ActivationFunction.Linear;

        public override double[] Forward(double[] inputs)
        {
            if (SubLayers.Count == 0)
                throw new InvalidOperationException("CompositeLayer has no sublayers.");

            // cache inputs/outputs per boundary for backward pass:
            _forwardCache = new double[SubLayers.Count + 1][];
            _forwardCache[0] = (double[])inputs.Clone();

            double[] current = inputs;
            for (int i = 0; i < SubLayers.Count; ++i)
            {
                current = SubLayers[i].Forward(current);
                _forwardCache[i + 1] = (double[])current.Clone();
            }

            return current;
        }

        public override double[] Backward(double[] input, double[] upstream, double[] dW, double[] db, double learningRate)
        {
            if (SubLayers.Count == 0)
                throw new InvalidOperationException("CompositeLayer has no sublayers.");

            // Validate gradient buffers sizes match concatenated sizes
            int totalW = SubLayers.Sum(l => l.Weights.Length);
            int totalB = SubLayers.Sum(l => l.Biases.Length);
            if (dW.Length != totalW || db.Length != totalB)
                throw new ArgumentException("Gradient arrays do not match composite internal sizes.");

            // We'll walk sublayers backward, handing each a slice of dW/db and the appropriate input for that layer.
            double[] grad = upstream;
            int wPos = totalW; // we will walk slices from the end to match backward order
            int bPos = totalB;

            for (int i = SubLayers.Count - 1; i >= 0; --i)
            {
                var layer = SubLayers[i];
                int layerWLen = layer.Weights.Length;
                int layerBLen = layer.Biases.Length;

                wPos -= layerWLen;
                bPos -= layerBLen;

                // slice dW/db for this layer
                double[] dWSlice = new double[layerWLen];
                Array.Copy(dW, wPos, dWSlice, 0, layerWLen);

                double[] dbSlice = new double[layerBLen];
                Array.Copy(db, bPos, dbSlice, 0, layerBLen);

                // input to this sublayer is cached at index i
                double[] layerInput = _forwardCache.Length > i ? _forwardCache[i] : input;

                // call backward on the sublayer; it returns gradient w.r.t its input
                double[] deltaPrev = layer.Backward(layerInput, grad, dWSlice, dbSlice, learningRate);

                // copy back computed gradients into the shared buffers
                Array.Copy(dWSlice, 0, dW, wPos, layerWLen);
                Array.Copy(dbSlice, 0, db, bPos, layerBLen);

                // set grad for next (previous) layer
                grad = deltaPrev;
            }

            return grad;
        }

        public override void UpdateWeights(double[] dW, double[] db, double learningRate, double l2)
        {
            if (SubLayers.Count == 0)
                return;

            int totalW = SubLayers.Sum(l => l.Weights.Length);
            int totalB = SubLayers.Sum(l => l.Biases.Length);
            if (dW.Length != totalW || db.Length != totalB)
                throw new ArgumentException("Gradient arrays do not match composite internal sizes.");

            int wPos = 0;
            int bPos = 0;
            foreach (var layer in SubLayers)
            {
                int wLen = layer.Weights.Length;
                int bLen = layer.Biases.Length;
                double[] dWSlice = new double[wLen];
                Array.Copy(dW, wPos, dWSlice, 0, wLen);
                double[] dbSlice = new double[bLen];
                Array.Copy(db, bPos, dbSlice, 0, bLen);

                layer.UpdateWeights(dWSlice, dbSlice, learningRate, l2);

                wPos += wLen;
                bPos += bLen;
            }
        }

        public override void Serialize(BinaryWriter writer)
        {
            // Composite marker and sublayer count, then for each sublayer write its type name and delegate serialization
            writer.Write("COMP"); // composite marker
            writer.Write(SubLayers.Count);
            foreach (var sub in SubLayers)
            {
                writer.Write(sub.GetType().FullName);
                // sublayer implementations write their own internal marker/data
                sub.Serialize(writer);
            }
        }

        public override void Deserialize(BinaryReader reader)
        {
            string marker = reader.ReadString();
            if (marker != "COMP")
                throw new Exception("Invalid composite layer marker.");

            int count = reader.ReadInt32();
            SubLayers = new List<ILayer>(count);

            for (int i = 0; i < count; ++i)
            {
                string subTypeName = reader.ReadString();
                Type? t = Type.GetType(subTypeName);
                if (t == null)
                    throw new Exception($"Unknown sublayer type: {subTypeName}");
                var instance = (ILayer)Activator.CreateInstance(t)!;
                instance.Deserialize(reader);
                SubLayers.Add(instance);
            }
        }

        protected void Add(ILayer layer)
        {
            SubLayers.Add(layer);
        }
    }
}
