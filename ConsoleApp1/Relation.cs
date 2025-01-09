using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections;
using System.Collections.Generic;

namespace CPU_GPU_System
{
    public class Relation
    {
        public string name;
        public string[] attributeNames;
        public List<dynamic> attributes;
        public Type[] types;

        public Relation(string name, string[] attributeNames)
        {
            this.name = name;
            this.attributeNames = new string[attributeNames.Length];
            this.types = new Type[attributeNames.Length];
            this.attributes = new List<dynamic>();

            for (int i = 0; i < attributeNames.Length; ++i)
            {
                this.attributeNames[i] = attributeNames[i];
            }
        }

        public Relation(string name, string[] attributeNames, List<dynamic> attributes) : this(name, attributeNames)
        {
            for (int i = 0; i < attributes.Count; ++i)
            {
                this.attributes.Add(attributes[i]);
                this.types[i] = attributes[i].GetAttrType();
            }
        }

        public dynamic this[int row, int column]
        {
            get
            {
                if (row < 0 || row >= this.attributes.Count || column < 0 || column > this.attributes[0].Count)
                {
                    throw new IndexOutOfRangeException("Index is out of range.");
                }

                return this.attributes[row][column];
            }

            set
            {
                if (row < 0 || row >= this.attributes.Count || column < 0 || column > this.attributes[0].Count)
                {
                    throw new IndexOutOfRangeException("Index is out of range.");
                }

                this.attributes[row][column] = value;
            }
        }

        public dynamic this[int index]
        {
            get
            {
                if (index < 0 || index >= this.Count)
                {
                    throw new IndexOutOfRangeException("Index is out of range.");
                }

                return attributes[index];
            }
        }

        public int Count
        {
            get { return this.attributes.Count; }
        }

        public ref string[] GetAttributeNames()
        {
            return ref this.attributeNames;
        }

        public void Print()
        {
            Console.WriteLine(this.name);
            for (int i = 0; i < this.attributeNames.Length; ++i)
            {
                Console.Write(this.attributeNames[i].PadLeft(10));
                Console.Write("\t");
            }
            Console.Write('\n');

            for (int i = 0; i < this.attributes[0].Count; ++i)
            {
                for (int j = 0; j < this.attributes.Count; ++j)
                {
                    if (this.attributes[j][i].HasValue > 0)
                    {
                        Console.Write(this.attributes[j][i].Value.ToString().PadLeft(10));
                    }
                    else
                    {
                        Console.Write("null".ToString().PadLeft(10));
                    }
                    Console.Write('\t');
                }
                Console.Write('\n');
            }

        }
    }

    public class _Attribute<T> where T : unmanaged
    {
        public string name;
        public T[] valuesCPU;
        public MemoryBuffer1D<T, Stride1D.Dense> valuesGPU;

        public _Attribute()
        {
            this.name = "new_attr";
        }

        public _Attribute(string name) : this()
        {
            this.name = name;
        }

        public _Attribute(string name, Accelerator accelerator, T[] data) : this(name)
        {
            this.valuesCPU = new T[data.Length];
            for (int i = 0; i < data.Length; ++i)
            {
                this.valuesCPU[i] = data[i];
            }

            this.valuesGPU = accelerator.Allocate1D(data);
        }

        ~_Attribute()
        {
            this.valuesGPU.Dispose();
        }

        public int Count
        {

            get { return valuesCPU.Length; }

        }

        public Type GetAttrType()
        {
            return typeof(T);
        }

        public ref MemoryBuffer1D<T, Stride1D.Dense> GetGPUValues()
        {
            return ref this.valuesGPU;
        }

        public void Synchronize()
        {
            this.valuesGPU.CopyToCPU(this.valuesCPU);
        }

        public T this[int index]
        {
            get
            {
                if (index < 0 || index >= valuesCPU.Length)
                {
                    throw new IndexOutOfRangeException("Index is out of range.");
                }

                return this.valuesCPU[index];
            }

            set
            {
                if (index < 0 || index >= valuesCPU.Length)
                {
                    throw new IndexOutOfRangeException("Index is out of range.");
                }

                this.valuesCPU[index] = value;
            }
        }
    }
}
