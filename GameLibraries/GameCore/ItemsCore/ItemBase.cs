using System;
using System.IO;
using TRW.CommonLibraries.Serialization;

namespace TRW.GameLibraries.GameCore
{
    [Serializable]
    public abstract class ItemBase : IBinarySerializable, IGameObject
    {
        public ItemBase(string name, string description, decimal weight)
        {
            this.Name = name;
            this.Description = description;
            this.Weight = weight;
        }

        public string Name { get; set; }
        public string Description { get; set; }
        public decimal Weight { get; set; }
        public bool IsPlayable => false;

        public virtual double X => throw new NotImplementedException();
        public virtual double Y => throw new NotImplementedException();
        public virtual double Width => throw new NotImplementedException();
        public virtual double Height => throw new NotImplementedException();

        public void WriteTo(BinaryWriter writer)
        {
            writer.Write(Name);
            writer.Write(Description);
            writer.Write(Weight);
            SerializeObject(writer);
        }
        public void ReadFrom(BinaryReader reader)
        {
            Name = reader.ReadString();
            Description = reader.ReadString();
            Weight = reader.ReadDecimal();
            DeserializeObject(reader);
        }
        protected abstract void SerializeObject(BinaryWriter writer);

        protected abstract void DeserializeObject(BinaryReader reader);

        public virtual void GameTimerTick()
        {

        }

        public byte[] ToByteArray()
        {
            using (MemoryStream ms = new MemoryStream())
            using (BinaryWriter bw = new BinaryWriter(ms))
            {
                WriteTo(bw);
                return ms.ToArray();
            }
        }

        public virtual bool CollidesWith(IGameObject otherGameObject)
        {
            throw new NotImplementedException();
        }
    }
}
