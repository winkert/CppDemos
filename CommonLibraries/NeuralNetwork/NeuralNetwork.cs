namespace TRW.CommonLibraries.NeuralNetwork
{
    public class NeuralNetwork
    {
        private readonly List<ILayer> _layers = [];
        private readonly Random _rand = new();
        private double[][]? _storedX;
        private double[][]? _storedY;

        public bool StoredPredictionsEnabled { get; set; } = false;
        public double StoredBlendAlpha { get; set; } = 0.5;

        public List<ILayer> Layers => _layers;
        public long InputSize => _layers.Count > 0 ? _layers[0].InputSize : 0;
        public long OutputSize => _layers.Count > 0 ? _layers.Last().OutputSize : 0;

        public NeuralNetwork() { }

        public void AddLayer(int inputSize, int outputSize, ActivationFunction activationFunction)
        {
            ILayer layer = new ActivationLayer(inputSize, outputSize, activationFunction);
            this.AddLayer(layer);
        }

        public void AddLayer(ILayer layer)
        {
            _layers.Add(layer);
        }
        /// <summary>
        /// Propagates the input values through the network layers and returns the resulting output.
        /// </summary>
        /// <remarks>The method processes the input values sequentially through all layers of the network.
        /// If stored predictions  are enabled and the necessary stored data is available, the output is blended with
        /// the stored predictions  before being returned.</remarks>
        /// <param name="inputs">An array of input values to be processed by the network. Cannot be null.</param>
        /// <returns>An array of output values produced by the network. If stored predictions are enabled and applicable,  the
        /// output may be blended with stored predictions.</returns>
        public double[] Forward(double[] inputs)
        {
            double[] output = inputs;
            foreach (ILayer layer in _layers)
            {
                output = layer.Forward(output);
            }

            if(StoredPredictionsEnabled && _storedX != null && _storedY != null)
            {
                return BlendStoredWithNetwork(inputs, output);
            }

            return output;
        }
        /// <summary>
        /// Trains the neural network using batch gradient descent on the provided dataset.
        /// </summary>
        /// <param name="X">Inputs</param>
        /// <param name="Y">Targets</param>
        /// <param name="learningRate">The learning rate used to scale the gradients during the update. Must be a positive value.</param>
        /// <param name="l2">The L2 regularization factor applied to the weights. Must be a non-negative value.</param>
        /// <param name="epochs">Iterations to run</param>
        /// <param name="shuffle">Randomize the indices</param>
        /// <returns></returns>
        public double TrainBatch(List<double[]> X, List<double[]> Y, double learningRate, double l2, int epochs, bool shuffle)
        {
            // If enabled, store the provided training data for later prediction/analysis
            if (StoredPredictionsEnabled) {
                StoreTrainingData(X, Y);
            }

            int nSamples = X.Count;
            double lastLoss = 0.0;
            for (int e = 0; e < epochs; e++)
            {
                List<int> indices = [.. Enumerable.Range(0, nSamples)];
                if (shuffle)
                {
                    indices = [.. indices.OrderBy(i => _rand.Next())];
                }
                double epochLoss = 0.0;
                int count = 0;
                foreach(int i in indices)
                {
                    epochLoss += TrainSample(X[i], Y[i], learningRate, l2);
                    count++;
                }
                lastLoss = epochLoss / Math.Max(1, count);
            }
            return lastLoss;
        }

        public void Serialize(string filePath)
        {
            using (var fs = new FileStream(filePath, FileMode.Create))
            using (var writer = new BinaryWriter(fs))
            {
                writer.Write("NNET"); // magic number
                writer.Write(1); // version
                writer.Write(_layers.Count);
                foreach (var layer in _layers)
                {
                    writer.Write(layer.GetType().FullName);
                    layer.Serialize(writer);
                }
            }
        }

        public void Deserialize(string filePath)
        {
            using (var fs = new FileStream(filePath, FileMode.Open))
            using (var reader = new BinaryReader(fs))
            {
                string magic = reader.ReadString();
                if (magic != "NNET")
                    throw new Exception("Invalid neural network file.");
                int version = reader.ReadInt32();
                if (version != 1)
                    throw new Exception("Unsupported neural network version.");
                int layerCount = reader.ReadInt32();
                _layers.Clear();
                for (int i = 0; i < layerCount; i++)
                {
                    string layerTypeName = reader.ReadString();
                    Type? layerType = Type.GetType(layerTypeName);
                    if (layerType == null)
                        throw new Exception($"Unknown layer type: {layerTypeName}");
                    var layer = (ILayer)Activator.CreateInstance(layerType)!;

                    layer.Deserialize(reader);
                    _layers.Add(layer);
                }
            }
        }

        internal double TrainSample(double[] x, double[] y, double learningRate, double l2)
        {
            List<double[]> activations = [x];
            foreach (ILayer layer in _layers)
            {
                activations.Add(layer.Forward(activations.Last()));
            }
            double[] yPred = activations.Last();
            double loss = ComputeLossAndDelta(yPred, y, out double[] delta);
            double[] upstream = delta;
            for (int li = _layers.Count - 1; li >= 0; li--)
            {
                ILayer layer = _layers[li];
                double[] inAct = activations[li];
                double[] dW = new double[layer.Weights.Length];
                double[] db = new double[layer.Biases.Length];
                double[] deltaPrev = layer.Backward(inAct, upstream, dW, db, learningRate);
                layer.UpdateWeights(dW, db, learningRate, l2);
                upstream = deltaPrev;
            }
            return loss;
        }

        private double ComputeLossAndDelta(double[] yPred, double[] yTrue, out double[] deltaOut)
        {
            ILayer lastLayer = _layers.Last();
            double loss = 0.0;
            deltaOut = new double[yPred.Length];
            if (lastLayer.ActivationFunction == ActivationFunction.Softmax)
            {
                double eps = 1e-12;
                for (int i = 0; i < yPred.Length; i++)
                {
                    loss += -yTrue[i] * Math.Log(yPred[i] + eps);
                    deltaOut[i] = yPred[i] - yTrue[i];
                }
            }
            else
            {
                for (int i = 0; i < yPred.Length; i++)
                {
                    double diff = yPred[i] - yTrue[i];
                    loss += 0.5 * diff * diff;
                    deltaOut[i] = diff;
                }
            }
            return loss;
        }

        private double[] BlendStoredWithNetwork(double[] inputs, double[] output)
        {
            int outDim = output.Length;
            // Prepare aggregated weighted sum
            double[] yAgg = new double[outDim];
            double weightSum = 0.0;
            const double eps = 1e-8;
            for (int si = 0; si < _storedX.Length; ++si)
            {
                double[] sx = _storedX[si];
                double[] sy = _storedY[si];
                if (sy.Length != outDim) continue; // mismatch, skip
                                                   // compute L2 distance between input and stored sample (use min length)
                double dist2 = 0.0;
                int len = Math.Min(inputs.Length, sx.Length);
                for (int k = 0; k < len; ++k)
                {
                    double d = inputs[k] - sx[k];
                    dist2 += d * d;
                }
                double dist = Math.Sqrt(dist2);
                double w = 1.0 / (dist + eps);
                weightSum += w;
                for (int j = 0; j < outDim; ++j) yAgg[j] += w * sy[j];
            }

            double[] storedPred = new double[outDim];
            if (weightSum > 0.0)
            {
                for (int j = 0; j < outDim; ++j) storedPred[j] = yAgg[j] / weightSum;
            }
            else
            {
                // no usable stored data, fallback to network
                return output;
            }

            // blend
            double alpha = StoredBlendAlpha;
            if (alpha < 0.0)
                alpha = 0.0;

            if (alpha > 1.0)
                alpha = 1.0;

            double[] blended = new double[outDim];
            for (int j = 0; j < outDim; ++j)
                blended[j] = alpha * output[j] + (1.0 - alpha) * storedPred[j];

            return blended;
        }

        private void StoreTrainingData(List<double[]> X, List<double[]> Y)
        {
            _storedX = new double[X.Count][];
            _storedY = new double[Y.Count][];
            for (int i = 0; i < X.Count; i++)
            {
                _storedX[i] = (double[])X[i].Clone();
            }
            for (int i = 0; i < Y.Count; i++)
            {
                _storedY[i] = (double[])Y[i].Clone();
            }
        }

        private void ClearStoredData()
        {
            _storedX = null;
            _storedY = null;
        }
    }
}
