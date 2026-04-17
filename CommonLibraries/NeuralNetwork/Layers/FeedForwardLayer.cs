using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TRW.CommonLibraries.NeuralNetwork
{
    public class FeedForwardLayer : CompositeLayer
    {
        public FeedForwardLayer(int modelDim, int hiddenDim)
            : base()
        {
            Add(new DenseLayer(modelDim, hiddenDim, ActivationFunction.ReLU));
            Add(new DenseLayer(hiddenDim, modelDim, ActivationFunction.Linear));
        }
    }

}
