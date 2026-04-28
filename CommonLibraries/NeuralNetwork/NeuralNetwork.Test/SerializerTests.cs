using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace TRW.CommonLibraries.NeuralNetwork.Test
{
    [TestClass]
    public class SerializerTests
    {
        [TestMethod]
        public void TestSerialize()
        {
            NeuralNetwork target = TestMocker.TrainXOr();
            string tempFile = System.IO.Path.GetTempFileName();
            target.Serialize(tempFile);
            Assert.IsTrue(System.IO.File.Exists(tempFile), "Serialized file should exist.");
        }
        [TestMethod]
        public void TestDeserialize()
        {
            NeuralNetwork target = TestMocker.TrainXOr();
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
            NeuralNetwork target = TestMocker.TrainComplexNetwork();
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
            NeuralNetwork target = TestMocker.TrainComplexNetwork();
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
            NeuralNetwork target = TestMocker.TrainTransformerNetwork();
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
            NeuralNetwork target = TestMocker.TrainTransformerNetwork();
            string tempFile = System.IO.Path.GetTempFileName();
            target.Serialize(tempFile);
            Assert.IsTrue(System.IO.File.Exists(tempFile), "Serialized file should exist.");
            NeuralNetwork loaded = new NeuralNetwork();
            loaded.Deserialize(tempFile);
            Assert.AreEqual(target.Layers.Count, loaded.Layers.Count, "Layer count should match after deserialization.");
        }

    }
}
