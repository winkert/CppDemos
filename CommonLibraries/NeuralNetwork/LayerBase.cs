namespace TRW.CommonLibraries.NeuralNetwork
{
    public abstract class LayerBase : ILayer
    {
        private static readonly Random Rand = new();
        private static readonly double SqrtTwoOverPi = Math.Sqrt(2 / Math.PI);

        public double[] Weights { get; set; }
        public double[] Biases { get; set; }
        protected double[] PostActivationVector { get; private set; }
        protected double[] PreActivationVector { get; private set; }

        public long InputSize => Weights.Length / Biases.Length;
        public long OutputSize => Biases.Length;

        public ActivationFunction ActivationFunction { get; set; }

        public LayerBase() 
        { 
            Weights = Array.Empty<double>();
            Biases = Array.Empty<double>();
            ActivationFunction = ActivationFunction.Linear;
            PreActivationVector = Array.Empty<double>();
            PostActivationVector = Array.Empty<double>();   
        }

        public LayerBase(long inputSize, long outputSize, ActivationFunction activationFunction)
        {
            Weights = new double[inputSize * outputSize];
            Biases = new double[outputSize];
            ActivationFunction = activationFunction;

            PreActivationVector = new double[Biases.Length];
            PostActivationVector = new double[Biases.Length];

            InitializeWeights();
        }

        public abstract double[] Forward(double[] inputs);
        public abstract double[] Backward(double[] input, double[] upstream, double[] dW, double[] db, double learningRate);
        public abstract void UpdateWeights(double[] dW, double[] db, double learningRate, double l2);

        protected void InitializeWeights()
        {
            // Use fan-in based scale and symmetric initialization around zero.
            double fanIn = Math.Max(1, InputSize);
            double scale = 1.0 / Math.Sqrt(fanIn); // Xavier-style default
            switch (ActivationFunction)
            {
                case ActivationFunction.ReLU:
                case ActivationFunction.LeakyReLU:
                    scale = Math.Sqrt(2.0 / fanIn); // He initialization
                    break;
            }

            for (int i = 0; i < Weights.Length; i++)
            {
                // sample in [-scale, +scale]
                Weights[i] = (Rand.NextDouble() * 2.0 - 1.0) * scale;
            }
            for (int i = 0; i < Biases.Length; i++)
            {
                Biases[i] = 0;
            }

        }

        #region Activation Functions
        public static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        public static double SigmoidDerivative(double a) => a * (1 - a);

        public static double Linear(double x) => x;
        public static double LinearDerivative(double x) => 1.0;

        public static double Tanh(double x) => Math.Tanh(x);
        public static double TanhDerivative(double x) => 1 - Math.Pow(Math.Tanh(x), 2);

        public static double ReLU(double x) => x > 0 ? x : 0.0;
        // ReLU derivative should be 1 for positive pre-activation, 0 for non-positive
        public static double ReLUDerivative(double x) => x > 0 ? 1.0 : 0.0;

        public static double LeakyReLU(double x, double alpha = 0.01) => x > 0 ? x : alpha * x;
        public static double LeakyReLUDerivative(double x, double alpha = 0.01) => x > 0 ? 1.0 : alpha;

        public static double Swish(double x) => x / (1.0 + Math.Exp(-x));
        public static double SwishDerivative(double x) => (1.0 - Math.Exp(-x));

        public static double HardSwish(double x)
        {
            double hs = Math.Max(0.0, Math.Min(1.0, 0.2 * x + 0.5));
            return x * hs;
        }
        public static double HardSwishDerivative(double x)
        {
            if (x <= -2.5) return 0.0;
            if (x >= 2.5) return 1.0;
            return 0.4 * x + 0.5;
        }

        public static double ELU(double x, double alpha = 1.0)=> x > 0 ? x : alpha * (Math.Exp(x) - 1.0);
        public static double ELUDerivative(double x, double alpha = 1.0) => x > 0 ? 1.0 : alpha * Math.Exp(x);


        public static double GELU(double x)
        {
            double inner = SqrtTwoOverPi * (x + 0.044715 * Math.Pow(x, 3));
            return 0.5 * x * (1.0 + Math.Tanh(inner));
        }
        public static double GELUDerivative(double x)
        {
            double x3 = x * x * x;
            double t = SqrtTwoOverPi * (x + 0.044715 * x3);
            double tanhT = Math.Tanh(t);

            double dt_dx = SqrtTwoOverPi * (1.0 + 3.0 * 0.044715 * x * x);
            double sech2 = 1.0 - tanhT * tanhT;

            return 0.5 * (1.0 + tanhT) + 0.5 * x * sech2 * dt_dx;
        }


        public static double[] Softmax(double[] inputs)
        {
            double max = double.MinValue;
            foreach (var val in inputs)
                if (val > max) max = val; // numerical stability

            double sumExp = 0.0;
            double[] expVals = new double[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                expVals[i] = Math.Exp(inputs[i] - max);
                sumExp += expVals[i];
            }

            double[] outputs = new double[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                outputs[i] = expVals[i] / sumExp;

            return outputs;
        }

        public static double[] SoftmaxDerivative(double[] outputs)
        {
            double[] derivatives = new double[outputs.Length];
            for (int i = 0; i < outputs.Length; i++)
                derivatives[i] = outputs[i] * (1 - outputs[i]);
            return derivatives;
        }

        #endregion

        protected double Activate(double x, ActivationFunction function)
        {
            return function switch
            {
                ActivationFunction.Sigmoid => Sigmoid(x),
                ActivationFunction.Linear => Linear(x),
                ActivationFunction.Tanh => Tanh(x),
                ActivationFunction.ReLU => ReLU(x),
                ActivationFunction.LeakyReLU => LeakyReLU(x),
                ActivationFunction.Swish => Swish(x),
                ActivationFunction.HardSwish => HardSwish(x),
                ActivationFunction.ELU => ELU(x),
                ActivationFunction.GELU => GELU(x),
                _ => x,
            };
        }

        protected double ActivateDerivative(double x, double a, ActivationFunction function)
        {
            return function switch
            {
                ActivationFunction.Sigmoid => SigmoidDerivative(a), // a = sigmoid(x)
                ActivationFunction.Linear => LinearDerivative(x),
                ActivationFunction.Tanh => TanhDerivative(a),
                ActivationFunction.ReLU => ReLUDerivative(x),
                ActivationFunction.LeakyReLU => LeakyReLUDerivative(x),
                ActivationFunction.Swish => SwishDerivative(x),
                ActivationFunction.HardSwish => HardSwishDerivative(x),
                ActivationFunction.ELU => ELUDerivative(x),
                ActivationFunction.GELU => GELUDerivative(x),
                _ => 1.0,
            };
        }

        protected double[] ActivateArray(double[] x, ActivationFunction function)
        {
            if (function == ActivationFunction.Softmax)
            {
                return Softmax(x);
            }

            double[] result = new double[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                result[i] = Activate(x[i], function);
            }
            return result;
        }

        public virtual void Serialize(System.IO.BinaryWriter writer)
        {
            writer.Write("LAY");
            writer.Write((int)ActivationFunction);
            writer.Write(Weights.Length);
            foreach (var w in Weights)
                writer.Write(w);
            writer.Write(Biases.Length);
            foreach (var b in Biases)
                writer.Write(b);
        }

        public virtual void Deserialize(System.IO.BinaryReader reader)
        {
            string t = reader.ReadString();
            // should be LAY
            ActivationFunction = (ActivationFunction)reader.ReadInt32();
            int weightsLength = reader.ReadInt32();
            Weights = new double[weightsLength];
            for (int i = 0; i < weightsLength; i++)
                Weights[i] = reader.ReadDouble();
            int biasesLength = reader.ReadInt32();
            Biases = new double[biasesLength];
            for (int i = 0; i < biasesLength; i++)
                Biases[i] = reader.ReadDouble();

            PreActivationVector = new double[Biases.Length];
            PostActivationVector = new double[Biases.Length];
        }

    }
}
