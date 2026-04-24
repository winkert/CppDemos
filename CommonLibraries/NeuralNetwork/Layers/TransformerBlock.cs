using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TRW.CommonLibraries.NeuralNetwork
{
    public class TransformerBlock : CompositeLayer
    {
        public TransformerBlock() : base() { }
        public TransformerBlock(int modelDim, int headCount, int ffHidden)
            :base()
        {
            Add(new LayerNormLayer(modelDim));
            Add(new MultiHeadAttentionLayer(modelDim, headCount));
            Add(new ResidualAddLayer(modelDim));

            Add(new LayerNormLayer(modelDim));
            Add(new FeedForwardLayer(modelDim, ffHidden));
            Add(new ResidualAddLayer(modelDim));
        }
    }

}
