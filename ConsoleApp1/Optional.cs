using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CPU_GPU_System
{
    public struct Optional<T> : IComparable<Optional<T>> where T : unmanaged, IComparable<T>
    {
        public T Value;
        public short HasValue;

        public Optional()
        {
            this.Value = default;
            this.HasValue = 0;
        }

        public Optional(T value)
        {
            this.Value = value;
            this.HasValue = 1;
        }

        public Optional(short hasValue)
        {
            this.Value = default;
            this.HasValue = hasValue;
        }

        public int CompareTo(Optional<T> compareWith)
        {
            return this.Value.CompareTo(compareWith.Value);
        }
    }
}
