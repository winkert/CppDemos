using System;
using System.Collections.Generic;
using System.Linq;
using TRW.CommonLibraries.NeuralNetwork;
using TRW.Games.Pong.GameObjects;

namespace TRW.Games.Pong.PongAI
{
    internal static class PongTrainer
    {
        internal static bool UseAIModel { get; set; } = false;

        private static string _trainingModelPath;
        internal static string TrainingModelPath
        {
            get
            {
                if (string.IsNullOrEmpty(_trainingModelPath))
                {
                    _trainingModelPath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "pongai.bin");
                }
                return _trainingModelPath;
            }
        }

        private static NeuralNetwork _pongAI;
        internal static NeuralNetwork PongAI
        {
            get
            {
                if (_pongAI == null)
                {
                    if (System.IO.File.Exists(TrainingModelPath))
                    {
                        _pongAI = new NeuralNetwork();
                        _pongAI.Deserialize(TrainingModelPath);
                    }
                    else
                    {
                        throw new System.IO.IOException("No AI Model Found For Pong");
                    }
                }
                return _pongAI;
            }
        }

        internal static bool TrainingMode { get; set; } = true;
        internal static List<PongGameState> TrainingData { get; } = new List<PongGameState>();
        internal static void AddTrainingData(Ball theBall, Paddle me, bool movingUp)
        {
            // training data comes from player paddle movement
            PongGameState gameState = new PongGameState
            {
                BallX = (int)theBall.Left,
                BallY = (int)theBall.Top,
                MyPaddleY = (int)me.Top,
                BallHeadingToMe = theBall.VelocityX > 0, // Assuming right paddle is the AI
                BallSpeed = (int)theBall.Speed,
                MoveUp = movingUp
            };
            TrainingData.Add(gameState);
        }

        internal static void AddTrainingData(PongGameState gameState)
        {
            if (TrainingMode)
            {
                TrainingData.Add(gameState);
            }
        }

        internal static void ApplyTraining()
        {
            if (!TrainingMode)
                return;

            if(System.IO.File.Exists(TrainingModelPath))
            {
                System.IO.File.Delete(TrainingModelPath);
            }

            // ideally, we would append training data and kind of retrain the model incrementally, but for simplicity, we'll just retrain from scratch each time
            NeuralNetwork model = new NeuralNetwork();
            model.AddLayer(new DenseLayer(5, 10, ActivationFunction.ReLU));
            model.AddLayer(new DenseLayer(10, 1, ActivationFunction.Sigmoid));

            // prepare training data
            List<double[]> inputs = TrainingData.Select(d => new double[] { d.BallX, d.BallY, d.MyPaddleY, d.BallHeadingToMe ? 1.0 : 0.0, d.BallSpeed }).ToList();
            List<double[]> outputs = TrainingData.Select(d => new double[] { d.MoveUp ? 1.0 : 0.0 }).ToList();

            model.TrainBatch(inputs, outputs, 0.01, 1e4, 1000, true);
            model.Serialize(TrainingModelPath);

            TrainingData.Clear();
        }

        internal static bool MovePaddle(Ball theBall, Paddle me)
        {
            PongGameState gameState = new PongGameState
            {
                BallX = (int)theBall.Left,
                BallY = (int)theBall.Top,
                MyPaddleY = (int)me.Top,
                BallHeadingToMe = theBall.VelocityX > 0, // Assuming right paddle is the AI
                BallSpeed = (int)theBall.Speed,
            };

            double[] inputs = new double[] { gameState.BallX, gameState.BallY, gameState.MyPaddleY, gameState.BallHeadingToMe ? 1.0 : 0.0, gameState.BallSpeed };
            double[] results = PongAI.Forward(inputs);
            return Convert.ToBoolean((int)results[0]);
        }
    }
}
