using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using TRW.CommonLibraries.NeuralNetwork;

namespace TRW.CommonLibraries.NeuralNetwork.Test
{
    [TestClass]
    public class SerializerTests
    {
        [TestMethod]
        public void TestSerialize()
        {
            NeuralNetwork target = TrainXOr();
            string tempFile = System.IO.Path.GetTempFileName();
            target.Serialize(tempFile);
            Assert.IsTrue(System.IO.File.Exists(tempFile), "Serialized file should exist.");
        }
        [TestMethod]
        public void TestDeserialize()
        {
            NeuralNetwork target = TrainXOr();
            string tempFile = System.IO.Path.GetTempFileName();
            target.Serialize(tempFile);
            Assert.IsTrue(System.IO.File.Exists(tempFile), "Serialized file should exist.");

            NeuralNetwork loaded = new NeuralNetwork();
            loaded.Deserialize(tempFile);
            Assert.AreEqual(target.Layers.Count, loaded.Layers.Count, "Layer count should match after deserialization.");

            // run tests and validate against known expected outputs
            double[][] X = {
                [0,0], [0,1], [1, 0], [1, 1]
            };
            double[][] Y = {
                [1, 0], [0,1], [0,1], [1,0] // class0 for (0,0) and (1,1), class1 for (0,1),(1,0)
            };

            for (int i = 0; i < X.Length; ++i)
            {
                double[] p = loaded.Forward(X[i]);
                Console.WriteLine($"Input [{X[i][0]},{X[i][1]}] -> probs [{p[0]}, {p[1]}]");

                // 1,0 if 0,0 or 1,1
                if (X[i][0] == X[i][1])
                {
                    Assert.AreEqual(1, p[0], 0.001, $"Expected ~1,0 for [{X[i][0]},{X[i][1]}], got [{p[0]}, {p[1]}]");
                }
                // 0,1 else
                else
                {
                    Assert.AreEqual(1, p[1], 0.001, $"Expected ~0,1 for [{X[i][0]},{X[i][1]}], got [{p[0]}, {p[1]}]");

                }
            }
        }

        [TestMethod]
        public void TestSerializeComplexNetwork()
        {
            NeuralNetwork target = TrainComplexNetwork();
            string tempFile = System.IO.Path.GetTempFileName();
            target.Serialize(tempFile);
            Assert.IsTrue(System.IO.File.Exists(tempFile), "Serialized file should exist.");
            NeuralNetwork loaded = new NeuralNetwork();
            loaded.Deserialize(tempFile);
            Assert.AreEqual(target.Layers.Count, loaded.Layers.Count, "Layer count should match after deserialization.");
            // run tests and validate against known expected outputs
            for (int i = 0; i < 8; ++i)
            {
                double[] raw = [(i & 1), ((i >> 1) & 1), ((i >> 2) & 1), ((i >> 3) & 1)];
                double[] pTarget = target.Forward(raw);
                double[] pLoaded = loaded.Forward(raw);
                Console.WriteLine($"Input [{raw[0]},{raw[1]},{raw[2]},{raw[3]}] -> target probs [{string.Join(", ", pTarget.Select(x => x.ToString("F6")))}], loaded probs [{string.Join(", ", pLoaded.Select(x => x.ToString("F6")))}]");
                // we won't assert exact values here since it's a more complex network, but we should see a clear pattern in the probabilities.
            }
        }
        [TestMethod]
        public void TestDeserializeComplexNetwork()
        {
            NeuralNetwork target = TrainComplexNetwork();
            string tempFile = System.IO.Path.GetTempFileName();
            target.Serialize(tempFile);
            Assert.IsTrue(System.IO.File.Exists(tempFile), "Serialized file should exist.");
            NeuralNetwork loaded = new NeuralNetwork();
            loaded.Deserialize(tempFile);
            Assert.AreEqual(target.Layers.Count, loaded.Layers.Count, "Layer count should match after deserialization.");
            // run tests and validate against known expected outputs
            for (int i = 0; i < 8; ++i)
            {
                double[] raw = [(i & 1), ((i >> 1) & 1), ((i >> 2) & 1), ((i >> 3) & 1)];
                double[] pTarget = target.Forward(raw);
                double[] pLoaded = loaded.Forward(raw);
                Console.WriteLine($"Input [{raw[0]},{raw[1]},{raw[2]},{raw[3]}] -> target probs [{string.Join(", ", pTarget.Select(x => x.ToString("F6")))}], loaded probs [{string.Join(", ", pLoaded.Select(x => x.ToString("F6")))}]");
                // we won't assert exact values here since it's a more complex network, but we should see a clear pattern in the probabilities.
            }
        }

        [TestMethod]
        public void TestSerializeTransformerNetwork()
        {
            NeuralNetwork target = TrainTransformerNetwork();
            string tempFile = System.IO.Path.GetTempFileName();
            target.Serialize(tempFile);
            Assert.IsTrue(System.IO.File.Exists(tempFile), "Serialized file should exist.");
            NeuralNetwork loaded = new NeuralNetwork();
            loaded.Deserialize(tempFile);
            Assert.AreEqual(target.Layers.Count, loaded.Layers.Count, "Layer count should match after deserialization.");
        }

        [TestMethod]
        public void TestDeserializeTransformerNetwork()
        {
            NeuralNetwork target = TrainTransformerNetwork();
            string tempFile = System.IO.Path.GetTempFileName();
            target.Serialize(tempFile);
            Assert.IsTrue(System.IO.File.Exists(tempFile), "Serialized file should exist.");
            NeuralNetwork loaded = new NeuralNetwork();
            loaded.Deserialize(tempFile);
            Assert.AreEqual(target.Layers.Count, loaded.Layers.Count, "Layer count should match after deserialization.");
        }

        private NeuralNetwork TrainXOr()
        {
            double[][] X = {
                [0,0], [0,1], [1, 0], [1, 1]
            };
            double[][] Y = {
                [1, 0], [0,1], [0,1], [1,0] // class0 for (0,0) and (1,1), class1 for (0,1),(1,0)
            };

            NeuralNetwork nn = new NeuralNetwork();
            nn.AddLayer(2, 8, ActivationFunction.LeakyReLU);
            nn.AddLayer(8, 8, ActivationFunction.LeakyReLU);
            nn.AddLayer(8, 8, ActivationFunction.LeakyReLU);
            nn.AddLayer(8, 2, ActivationFunction.Softmax);

            double finalLoss = nn.TrainBatch(X.ToList(), Y.ToList(), 0.1, 1e-4, 5000, true);
            Console.WriteLine($"Training done. Loss={finalLoss}\n");

            /*
             * Training done. Loss=9.77151549912004E-05
             *
             * Input [0,0] -> probs [0.9998943560801821, 0.0001056439198179512]
             * Input [0,1] -> probs [6.67864108051393E-05, 0.9999332135891948]
             * Input [1,0] -> probs [0.00011299234525851499, 0.9998870076547416]
             * Input [1,1] -> probs [0.9998949630207341, 0.00010503697926600332]
             * 
             */

            return nn;
        }

        private NeuralNetwork TrainComplexNetwork()
        {
            // Build a small but varied network using CompositeLayer (Dense + Activation) and a final softmax output.
            NeuralNetwork nn = new NeuralNetwork();

            // Composite: 4 -> 6 -> 6 (LeakyReLU), then another composite 6 -> 6 -> 6 (ReLU)
            nn.AddLayer(new CompositeLayer(4, 6, 6, ActivationFunction.LeakyReLU));
            nn.AddLayer(new CompositeLayer(6, 6, 6, ActivationFunction.ReLU));

            // Final classifier layer: 6 -> 3 with softmax
            nn.AddLayer(6, 3, ActivationFunction.Softmax);

            // Prepare dataset: all 4-bit binary vectors (16 samples), class = sum(bits) % 3 (3 classes)
            var dataset = GenerateParityDataset(4, 3); // 4-bit parity dataset (16 samples, 3 classes)
            var X = dataset.Item1;
            var Y = dataset.Item2;

            double finalLoss = nn.TrainBatch(X, Y, 0.05, 1e-4, 2000, true);
            Console.WriteLine($"Complex network training done. Loss={finalLoss}\n");

            // show a few predictions
            for (int i = 0; i < 8; ++i)
            {
                double[] p = nn.Forward(X[i]);
                Console.WriteLine($"Input [{string.Join(",", X[i])}] -> probs [{string.Join(", ", p.Select(x => x.ToString("F6")))}]");
            }

            return nn;
        }

        private NeuralNetwork TrainTransformerNetwork()
        {
            NeuralNetwork nn = new NeuralNetwork();
            nn.AddLayer(new TransformerBlock(4, 2, 4)); // 2 heads, d_model=4
            nn.AddLayer(new CompositeLayer(4, 8, 4, ActivationFunction.ReLU)); // feedforward

            // final classifier to match parity dataset (2 classes)
            nn.AddLayer(4, 2, ActivationFunction.Softmax);

            // need to mock data to train it - hard coded values that can be tested against after deserialization
            var dataset = GenerateParityDataset(4); // 4-bit parity dataset (16 samples, 2 classes)
            double finalLoss = nn.TrainBatch(dataset.Item1, dataset.Item2, 0.05, 1e-4, 2000, true);
            Console.WriteLine($"Transformer network training done. Loss={finalLoss}\n");

            // show a few predictions
            for (int i = 0; i < 8; ++i)
            {
                double[] p = nn.Forward(dataset.Item1[i]);
                Console.WriteLine($"Input [{string.Join(",", dataset.Item1[i])}] -> probs [{string.Join(", ", p.Select(x => x.ToString("F6")))}]");
            }


            return nn;
        }

        private Tuple<List<double[]>, List<double[]>> GenerateParityDataset(int bitCount, int classCount = 2)
        {
            var X = new List<double[]>();
            var Y = new List<double[]>();
            int sampleCount = 1 << bitCount; // 2^bitCount samples
            for (int i = 0; i < sampleCount; ++i)
            {
                double[] v = new double[bitCount];
                int sum = 0;
                for (int b = 0; b < bitCount; ++b)
                {
                    v[b] = ((i >> b) & 1);
                    sum += (int)v[b];
                }
                int label = sum % classCount; // distribute counts across requested number of classes
                double[] o = new double[classCount];
                o[label] = 1.0;
                X.Add(v);
                Y.Add(o);
            }
            return Tuple.Create(X, Y);
        }
    }
}
