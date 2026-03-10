namespace TRW.CommonLibraries.NeuralNetwork
{
    public interface ILayer
    {
        double[] Weights { get; set; }
        double[] Biases { get; set; }
        long InputSize { get; }
        long OutputSize { get; }
        ActivationFunction ActivationFunction { get; }
        /// <summary>
        /// Computes the output of the model for the given input values.
        /// </summary>
        /// <param name="inputs">An array of input values to be processed by the model. The array must not be null and should match the
        /// expected input size of the model.</param>
        /// <returns>An array of output values produced by the model. The size and meaning of the output depend on the model's
        /// configuration.</returns>
        double[] Forward(double[] inputs);
        /// <summary>
        /// Performs the backward pass of the layer, calculating gradients and updating weights and biases.
        /// </summary>
        /// <param name="input">The input values from the forward pass.</param>
        /// <param name="upstream">The gradient values propagated from the next layer in the network.</param>
        /// <param name="dW">An array to store the computed gradients of the weights.</param>
        /// <param name="db">An array to store the computed gradients of the biases.</param>
        /// <param name="learningRate">The learning rate used to update the weights and biases.</param>
        /// <returns>An array representing the gradient values to propagate to the previous layer.</returns>
        double[] Backward(double[] input, double[] upstream, double[] dW, double[] db, double learningRate);
        /// <summary>
        /// Updates the weights and biases of the model using the provided gradients and learning rate.
        /// </summary>
        /// <param name="dW">An array of gradients for the weights. Each element represents the gradient for a corresponding weight.</param>
        /// <param name="db">An array of gradients for the biases. Each element represents the gradient for a corresponding bias.</param>
        /// <param name="learningRate">The learning rate used to scale the gradients during the update. Must be a positive value.</param>
        /// <param name="l2">The L2 regularization factor applied to the weights. Must be a non-negative value.</param>
        void UpdateWeights(double[] dW, double[] db, double learningRate, double l2);
        void Serialize(System.IO.BinaryWriter writer);
        void Deserialize(System.IO.BinaryReader reader);

    }
}
