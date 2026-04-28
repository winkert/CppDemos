using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;

namespace TRW.CommonLibraries.NeuralNetwork.Test
{
    [TestClass]
    public sealed class NeuralNetworkTests
    {
        [TestMethod]
        public void TestXorProbability ()
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

            Assert.IsTrue(finalLoss > 0);
            Assert.AreEqual(0, (int)finalLoss, $"finalLoss [{finalLoss}]");

            for (int i = 0; i < X.Length; ++i)
            {
                double[] p = nn.Forward(X[i]);
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

            /*
             * Training done. Loss=9.77151549912004E-05
             *
             * Input [0,0] -> probs [0.9998943560801821, 0.0001056439198179512]
             * Input [0,1] -> probs [6.67864108051393E-05, 0.9999332135891948]
             * Input [1,0] -> probs [0.00011299234525851499, 0.9998870076547416]
             * Input [1,1] -> probs [0.9998949630207341, 0.00010503697926600332]
             * 
             */

        }

        [TestMethod]
        public void TestTrainer()
        {
            Trainer trainer = new Trainer();
            trainer.AddLayer(2, ActivationFunction.Tanh);
            trainer.AddLayer(2, ActivationFunction.Linear);
            trainer.AddLayer(2, ActivationFunction.Linear);
            trainer.AddLayer(2, ActivationFunction.Linear);
            trainer.AddLayer(2, ActivationFunction.Linear);
            trainer.AddLayer(2, 1, ActivationFunction.Tanh);

            var data = TestMocker.GenerateZipCodeDataSet();
            var X = data.Item1;
            var Y = data.Item2;

            trainer.Train(X, Y, epochs: 5000, learningRate: 0.01);

            List<double[]> testInputs = new List<double[]>
            {
                new double[] {2,63121},
                new double[] {6,63121},
                new double[] {1,20012}
            };

            List<double> expected = [2, 4, 4];

            for(int i = 0; i < testInputs.Count; i++)
            {
                double[] output = trainer.Predict(testInputs[i]);
                Console.WriteLine($"Input: [{string.Join(", ", testInputs[i])}] => Predicted: {output[0]}, Expected: {expected[i]}");
            }
            /*
             * Input: [2, 63121] => Predicted: 1.7899772331466752, Expected: 2
             * Input: [6, 63121] => Predicted: 3.63250721118834, Expected: 4
             * Input: [1, 20012] => Predicted: 5.63218631181756, Expected: 4
             */
        }

        [TestMethod]
        public void TestTeachingAddition()
        {
            double scale = 10.0;
            var dataset = TestMocker.MakeAddDataset(2000, scale, true, new Random(42));

            // Example: Addition problem
            double[][] inputs = dataset.X;
            double[][] targets = dataset.Y;

            Trainer target = new Trainer();
            target.AddLayer(2, 4, ActivationFunction.Linear);
            target.AddLayer(4, 8, ActivationFunction.Linear);
            target.AddLayer(8, ActivationFunction.Linear);
            target.AddLayer(8, ActivationFunction.Linear);
            target.AddLayer(8, 4, ActivationFunction.Linear);
            target.AddLayer(4, 1, ActivationFunction.Linear);

            target.Train(inputs.ToList(), targets.ToList(), 1000);
            double[][] testInputs = [
                [1,2], // 3
                [3,4], // 7
                [5,7], // 12
                [10,5], // 15
                [10,15], // 25
                [-5,10],  // 5
                [0.5,0.25] // 0.75
            ];
            double[] expected = [3, 7, 12, 15, 25, 5, 0.75];

            for (int i = 0; i < testInputs.Length; i++)
            {
                var actual = target.Predict(testInputs[i]);
                Assert.AreEqual(expected[i], actual[0], 0.1, $"Addition test failed for input {testInputs[i][0]} + {testInputs[i][1]}");
                Console.WriteLine($"{testInputs[i][0]} + {testInputs[i][1]} = Predicted: {actual[0]}, Expected: {expected[i]}");
            }
            /*
             * 1 + 2 = Predicted: 2.994314391093891, Expected: 3
             * 3 + 4 = Predicted: 6.986417010436611, Expected: 7
             * 5 + 7 = Predicted: 11.976568861724326, Expected: 12
             * 10 + 5 = Predicted: 14.970480786466249, Expected: 15
             * 10 + 15 = Predicted: 24.95097310591608, Expected: 25
             * -5 + 10 = Predicted: 4.990695780295491, Expected: 5
             * 0.5 + 0.25 = Predicted: 0.748727196326981, Expected: 0.75
             */
        }

        [TestMethod, Ignore]
        public void TestTeachingMultiplication()
        {
            double scale = 10.0;
            var dataset = TestMocker.MakeMulDataset(1500, scale, true, new Random(42));

            // Example: Addition problem
            double[][] inputs = dataset.X;
            double[][] targets = dataset.Y;

            Trainer target = new Trainer();
            target.AddLayer(2, 16, ActivationFunction.Linear);
            target.AddLayer(16, ActivationFunction.ReLU);
            target.AddLayer(16, ActivationFunction.ReLU);
            target.AddLayer(16, 1, ActivationFunction.Linear);

            target.Train(inputs.ToList(), targets.ToList(), 2000);
            double[][] testInputs = [
                [1,2], // 2
                [3,4], // 12
                [5,7], // 35
                [1.5,5], // 7.5
                [10,1.5], // 15
                [-5,10],  // -50
                [0.5,0.25] // 0.125
            ];
            double[] expected = [2, 12, 35, 7.5, 15, -50, 0.125];

            for (int i = 0; i < testInputs.Length; i++)
            {
                var actual = target.Predict(testInputs[i]);
                //Assert.AreEqual(expected[i], actual[0], 1, $"Multiplication test failed for input {testInputs[i][0]} * {testInputs[i][1]}");
                Console.WriteLine($"{testInputs[i][0]} * {testInputs[i][1]} = Predicted: {actual[0]}, Expected: {expected[i]}");
            }
            /* (Still working on this training setup)
             * 1 * 2 = Predicted: 4.24024072703773, Expected: 2 
             *      (I've gotten closer to 1.5 with other runs/setups)
             * 3 * 4 = Predicted: 13.801409156563295, Expected: 12
             * 5 * 7 = Predicted: 36.6697968527877, Expected: 35
             * 1.5 * 5 = Predicted: 7.536694320937912, Expected: 7.5
             * 10 * 1.5 = Predicted: 16.42776505389557, Expected: 15
             * -5 * 10 = Predicted: -48.14002322885127, Expected: -50
             * 0.5 * 0.25 = Predicted: 1.7736497703625105, Expected: 0.125
             *      (this one has had some wildly different results between negative numbers and things over 2)
             */
        }


    }
}
