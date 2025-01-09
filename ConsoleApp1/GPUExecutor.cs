using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;

namespace CPU_GPU_System
{
    enum Comparator
    {
        less,
        less_eq,
        greater,
        greater_eq,
        equal,
        not_equal
    }

    internal class GPUExecutor
    {
        public Context context;
        public Device device;
        public Accelerator accelerator;
        public Dictionary<Type, Delegate> projectionKernels;
        public Dictionary<Type, Delegate> selectionMaskKernels;
        public Dictionary<Type, Delegate> selectionKernels;
        public Dictionary<Type, Delegate> innerJoinMaskKernels;
        public Dictionary<Type, Delegate> innerJoinKernels;

        public GPUExecutor()
        {
            this.context = Context.CreateDefault();
            this.device = this.context.GetPreferredDevice(preferCPU: true);
            this.accelerator = this.device.CreateAccelerator(this.context);

            this.projectionKernels = new Dictionary<Type, Delegate>();
            this.selectionMaskKernels = new Dictionary<Type, Delegate>();
            this.selectionKernels = new Dictionary<Type, Delegate>();
            this.innerJoinMaskKernels = new Dictionary<Type, Delegate>();
            this.innerJoinKernels = new Dictionary<Type, Delegate>();

            this.LoadKernels();

        }

        ~GPUExecutor()
        {
            this.accelerator.Dispose();
            this.context.Dispose();
        }

        void LoadKernels()
        {
            Action<Index1D, ArrayView1D<Optional<int>, Stride1D.Dense>, short> loadedProjKernelInt = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Optional<int>, Stride1D.Dense>, short>(ProjectionKernel);
            Action<Index1D, ArrayView1D<Optional<float>, Stride1D.Dense>, short> loadedProjKernelFloat = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Optional<float>, Stride1D.Dense>, short>(ProjectionKernel);
            Action<Index1D, ArrayView1D<Optional<double>, Stride1D.Dense>, short> loadedProjKernelDouble = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Optional<double>, Stride1D.Dense>, short>(ProjectionKernel);

            this.projectionKernels[typeof(Optional<int>)] = loadedProjKernelInt;
            this.projectionKernels[typeof(Optional<float>)] = loadedProjKernelFloat;
            this.projectionKernels[typeof(Optional<double>)] = loadedProjKernelDouble;


            Action<Index1D, ArrayView1D<Optional<int>, Stride1D.Dense>, ArrayView1D<short, Stride1D.Dense>, Comparator, Optional<int>> loadedSelMKernelInt = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Optional<int>, Stride1D.Dense>, ArrayView1D<short, Stride1D.Dense>, Comparator, Optional<int>>(SelectionMaskKernel);
            Action<Index1D, ArrayView1D<Optional<float>, Stride1D.Dense>, ArrayView1D<short, Stride1D.Dense>, Comparator, Optional<float>> loadedSelMKernelFloat = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Optional<float>, Stride1D.Dense>, ArrayView1D<short, Stride1D.Dense>, Comparator, Optional<float>>(SelectionMaskKernel);
            Action<Index1D, ArrayView1D<Optional<double>, Stride1D.Dense>, ArrayView1D<short, Stride1D.Dense>, Comparator, Optional<double>> loadedSelMKernelDouble = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Optional<double>, Stride1D.Dense>, ArrayView1D<short, Stride1D.Dense>, Comparator, Optional<double>>(SelectionMaskKernel);

            this.selectionMaskKernels[typeof(Optional<int>)] = loadedSelMKernelInt;
            this.selectionMaskKernels[typeof(Optional<float>)] = loadedSelMKernelFloat;
            this.selectionMaskKernels[typeof(Optional<double>)] = loadedSelMKernelDouble;


            Action<Index1D, ArrayView1D<Optional<int>, Stride1D.Dense>, ArrayView1D<short, Stride1D.Dense>> loadedSelKernelInt = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Optional<int>, Stride1D.Dense>, ArrayView1D<short, Stride1D.Dense>>(SelectionKernel);
            Action<Index1D, ArrayView1D<Optional<float>, Stride1D.Dense>, ArrayView1D<short, Stride1D.Dense>> loadedSelKernelFloat = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Optional<float>, Stride1D.Dense>, ArrayView1D<short, Stride1D.Dense>>(SelectionKernel);
            Action<Index1D, ArrayView1D<Optional<double>, Stride1D.Dense>, ArrayView1D<short, Stride1D.Dense>> loadedSelKernelDouble = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Optional<double>, Stride1D.Dense>, ArrayView1D<short, Stride1D.Dense>>(SelectionKernel);

            this.selectionKernels[typeof(Optional<int>)] = loadedSelKernelInt;
            this.selectionKernels[typeof(Optional<float>)] = loadedSelKernelFloat;
            this.selectionKernels[typeof(Optional<double>)] = loadedSelKernelDouble;

            Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>> loadedInnJoinMKernelInt = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(InnerJoinMaskKernel);
            Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>> loadedInnJoinMKernelFloat = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(InnerJoinMaskKernel);
            Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>> loadedInnJoinMKernelDouble = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(InnerJoinMaskKernel);

            this.innerJoinMaskKernels[typeof(int)] = loadedInnJoinMKernelInt;
            this.innerJoinMaskKernels[typeof(float)] = loadedInnJoinMKernelFloat;
            this.innerJoinMaskKernels[typeof(double)] = loadedInnJoinMKernelDouble;


            Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int> loadedInnJoinKernelInt = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int>(InnerJoinKernel);
            Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int> loadedInnJoinKernelFloat = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int>(InnerJoinKernel);
            Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int> loadedInnJoinKernelDouble = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int>(InnerJoinKernel);

            this.innerJoinKernels[typeof(int)] = loadedInnJoinKernelInt;
            this.innerJoinKernels[typeof(float)] = loadedInnJoinKernelFloat;
            this.innerJoinKernels[typeof(double)] = loadedInnJoinKernelDouble;
        }

        static void ProjectionKernel<T>(Index1D index, ArrayView1D<T, Stride1D.Dense> data, short flag) where T : unmanaged
        {
            if (flag == 0)
            {
                data[index] = new T();
            }
        }

        static void SelectionMaskKernel<T>(Index1D index, ArrayView1D<T, Stride1D.Dense> data, ArrayView1D<short, Stride1D.Dense> mask, Comparator comparator, T compareWith) where T : unmanaged, IComparable<T>
        {

            switch (comparator)
            {
                case Comparator.less:
                    if (data[index].CompareTo(compareWith) < 0 && mask[index] == 1)
                    {
                        mask[index] = 1;
                    }
                    else
                    {
                        mask[index] = 0;
                    }
                    break;

                case Comparator.less_eq:
                    if (data[index].CompareTo(compareWith) <= 0 && mask[index] == 1)
                    {
                        mask[index] = 1;
                    }
                    else
                    {
                        mask[index] = 0;
                    }
                    break;

                case Comparator.greater:
                    if (data[index].CompareTo(compareWith) > 0 && mask[index] == 1)
                    {
                        mask[index] = 1;
                    }
                    else
                    {
                        mask[index] = 0;
                    }
                    break;

                case Comparator.greater_eq:
                    if (data[index].CompareTo(compareWith) >= 0 && mask[index] == 1)
                    {
                        mask[index] = 1;
                    }
                    else
                    {
                        mask[index] = 0;
                    }
                    break;

                case Comparator.equal:
                    if (data[index].CompareTo(compareWith) == 0 && mask[index] == 1)
                    {
                        mask[index] = 1;
                    }
                    else
                    {
                        mask[index] = 0;
                    }
                    break;

                case Comparator.not_equal:
                    if (data[index].CompareTo(compareWith) != 0 && mask[index] == 1)
                    {
                        mask[index] = 1;
                    }
                    else
                    {
                        mask[index] = 0;
                    }
                    break;
            }
        }

        static void SelectionKernel<T>(Index1D index, ArrayView1D<T, Stride1D.Dense> data, ArrayView1D<short, Stride1D.Dense> mask) where T : unmanaged
        {
            if (mask[index] == 0)
            {
                data[index] = new T();
            }
        }

        static void InnerJoinMaskKernel<T>(Index1D index, ArrayView1D<T, Stride1D.Dense> data1, ArrayView1D<T, Stride1D.Dense> data2, ArrayView1D<int, Stride1D.Dense> mask) where T : unmanaged, IComparable<T>
        {
            for (int i = 0; i < data2.Length; ++i)
            {
                if (data1[index].CompareTo(data2[i]) == 0)
                {
                    mask[index] = i;
                    break;
                }
            }
        }

        static void InnerJoinKernel<T>(Index1D index, ArrayView1D<T, Stride1D.Dense> data, ArrayView1D<T, Stride1D.Dense> addedAttrs, ArrayView1D<int, Stride1D.Dense> mask, int flag) where T : unmanaged
        {
            if (mask[index] == -1)
            {
                data[index] = new T();
            }
            else if (flag > 0)
            {
                data[index] = addedAttrs[mask[index]];
            }
        }

        void ExecuteInnerJoinMaskKernel<T>(MemoryBuffer1D<T, Stride1D.Dense> data1, MemoryBuffer1D<T, Stride1D.Dense> data2, MemoryBuffer1D<int, Stride1D.Dense> mask) where T : unmanaged, IComparable<T>
        {
            var kernelDelegate = (Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>)this.innerJoinMaskKernels[typeof(T)];

            this.accelerator.Synchronize();
            kernelDelegate((Index1D)data1.Length, data1.View, data2.View, mask.View);
            this.accelerator.Synchronize();
        }

        void ExecuteInnerJoinKernel<T>(MemoryBuffer1D<T, Stride1D.Dense> data, MemoryBuffer1D<T, Stride1D.Dense> addedAttrs, MemoryBuffer1D<int, Stride1D.Dense> mask, int flag) where T : unmanaged
        {
            var kernelDelegate = (Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int>)this.innerJoinKernels[typeof(T)];

            this.accelerator.Synchronize();
            kernelDelegate((Index1D)data.Length, data.View, addedAttrs.View, mask.View, flag);
            this.accelerator.Synchronize();
        }

        void ExecuteSelectionMaskKernel<T>(MemoryBuffer1D<T, Stride1D.Dense> data, MemoryBuffer1D<short, Stride1D.Dense> mask, Comparator comparator, T compareWith) where T : unmanaged, IComparable<T>
        {
            var kernelDelegate = (Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<short, Stride1D.Dense>, Comparator, T>)this.selectionMaskKernels[typeof(T)];

            this.accelerator.Synchronize();
            kernelDelegate((Index1D)data.Length, data.View, mask.View, comparator, compareWith);
            this.accelerator.Synchronize();
        }

        void ExecuteSelectionKernel<T>(MemoryBuffer1D<T, Stride1D.Dense> data, MemoryBuffer1D<short, Stride1D.Dense> mask) where T : unmanaged
        {
            var kernelDelegate = (Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<short, Stride1D.Dense>>)this.selectionKernels[typeof(T)];

            this.accelerator.Synchronize();
            kernelDelegate((Index1D)data.Length, data.View, mask.View);
            this.accelerator.Synchronize();
        }

        void ExecuteProjectionKernel<T>(MemoryBuffer1D<T, Stride1D.Dense> data, short mask) where T : unmanaged
        {
            var kernelDelegate = (Action<Index1D, ArrayView1D<T, Stride1D.Dense>, short>)this.projectionKernels[typeof(T)];

            this.accelerator.Synchronize();
            kernelDelegate((Index1D)data.Length, data.View, mask);
            this.accelerator.Synchronize();

        }

        public void Projection(ref Relation relation, ref string[] headers)
        {
            short[] mask = new short[relation.Count];

            string[] attributeNames = relation.GetAttributeNames();

            for (int i = 0; i < attributeNames.Length; ++i)
            {
                for (int j = 0; j < headers.Length; ++j)
                {
                    if (attributeNames[i] == headers[j])
                    {
                        mask[i] = 1;
                        break;
                    }

                    mask[i] = 0;
                }
            }

            for (int i = 0; i < mask.Length; ++i)
            {
                if (relation.types[i] == typeof(Optional<int>))
                {
                    var attr = (_Attribute<Optional<int>>)relation.attributes[i];
                    this.ExecuteProjectionKernel<Optional<int>>(attr.GetGPUValues(), mask[i]);
                    attr.Synchronize();
                }
                else if (relation.types[i] == typeof(Optional<float>))
                {
                    var attr = (_Attribute<Optional<float>>)relation.attributes[i];
                    this.ExecuteProjectionKernel<Optional<float>>(attr.GetGPUValues(), mask[i]);
                    attr.Synchronize();
                }
                else if (relation.types[i] == typeof(Optional<double>))
                {
                    var attr = (_Attribute<Optional<double>>)relation.attributes[i];
                    this.ExecuteProjectionKernel<Optional<double>>(attr.GetGPUValues(), mask[i]);
                    attr.Synchronize();
                }
            }
        }

        public void Selection(ref Relation relation, ref string[] headers, Comparator[] comparator, dynamic[] compareWith)
        {
            short[] m = new short[relation.attributes[0].Count];
            for (int i = 0; i < m.Length; ++i)
            {
                m[i] = 1;
            }
            MemoryBuffer1D<short, Stride1D.Dense> mask = this.accelerator.Allocate1D<short>(m);

            for (int i = 0; i < headers.Length; ++i)
            {
                for (int j = 0; j < relation.Count; ++j)
                {
                    if (headers[i] == relation.attributeNames[j])
                    {
                        if (relation.types[j] == typeof(Optional<int>))
                        {
                            var attr = (_Attribute<Optional<int>>)relation.attributes[j];
                            this.ExecuteSelectionMaskKernel<Optional<int>>(attr.GetGPUValues(), mask, comparator[i], compareWith[i]);
                        }
                        else if (relation.types[j] == typeof(Optional<float>))
                        {
                            var attr = (_Attribute<Optional<float>>)relation.attributes[j];
                            this.ExecuteSelectionMaskKernel<Optional<float>>(attr.GetGPUValues(), mask, comparator[i], compareWith[i]);
                        }
                        else if (relation.types[j] == typeof(Optional<double>))
                        {
                            var attr = (_Attribute<Optional<double>>)relation.attributes[j];
                            this.ExecuteSelectionMaskKernel<Optional<double>>(attr.GetGPUValues(), mask, comparator[i], compareWith[i]);
                        }
                        break;
                    }
                }
            }

            for (int i = 0; i < relation.Count; ++i)
            {
                if (relation.types[i] == typeof(Optional<int>))
                {
                    var attr = (_Attribute<Optional<int>>)relation.attributes[i];
                    this.ExecuteSelectionKernel<Optional<int>>(attr.GetGPUValues(), mask);
                    attr.Synchronize();
                }
                else if (relation.types[i] == typeof(Optional<float>))
                {
                    var attr = (_Attribute<Optional<float>>)relation.attributes[i];
                    this.ExecuteSelectionKernel<Optional<float>>(attr.GetGPUValues(), mask);
                    attr.Synchronize();
                }
                else if (relation.types[i] == typeof(Optional<double>))
                {
                    var attr = (_Attribute<Optional<double>>)relation.attributes[i];
                    this.ExecuteSelectionKernel<Optional<double>>(attr.GetGPUValues(), mask);
                    attr.Synchronize();
                }
            }
        }

        /*public ref Relation InnerJoin(ref Relation relation1, ref Relation relation2, string header)
        {

            int[] m = new int[relation1.attributes[0].Count];
            for (int i = 0; i < m.Length; ++i)
            {
                m[i] = -1;
            }
            MemoryBuffer1D<int, Stride1D.Dense> mask = this.accelerator.Allocate1D<int>(m);


            int index1 = -1;
            int index2 = -1;

            for (int i = 0; i < relation1.Count; ++i)
            {
                if (relation1.attributeNames[i] == header)
                {
                    index1 = i;
                    break;
                }
            }

            for (int i = 0; i < relation2.Count; ++i)
            {
                if (relation2.attributeNames[i] == header)
                {
                    index2 = i;
                    break;
                }
            }

            if (relation1.types[index1] == typeof(int))
            {
                var attr1 = (_Attribute<int>)relation1.attributes[index1];
                var attr2 = (_Attribute<int>)relation2.attributes[index2];
                this.ExecuteInnerJoinMaskKernel<int>(attr1.GetGPUValues(), attr2.GetGPUValues(), mask);
            }
            else if (relation1.types[index1] == typeof(float))
            {
                var attr1 = (_Attribute<float>)relation1.attributes[index1];
                var attr2 = (_Attribute<float>)relation2.attributes[index2];
                this.ExecuteInnerJoinMaskKernel<float>(attr1.GetGPUValues(), attr2.GetGPUValues(), mask);
            }
            else if (relation1.types[index1] == typeof(double))
            {
                var attr1 = (_Attribute<double>)relation1.attributes[index1];
                var attr2 = (_Attribute<double>)relation2.attributes[index2];
                this.ExecuteInnerJoinMaskKernel<double>(attr1.GetGPUValues(), attr2.GetGPUValues(), mask);
            }

            for (int i = 0; i < relation1.Count; ++i)
            {
                if (relation1.types[i] == typeof(int))
                {
                    var attr = (_Attribute<int>)relation1.attributes[i];
                    this.ExecuteInnerJoinKernel<int>(attr.GetGPUValues(), mask);
                    attr.Synchronize();
                }
                else if (relation1.types[i] == typeof(float))
                {
                    var attr = (_Attribute<float>)relation1.attributes[i];
                    this.ExecuteInnerJoinKernel<float>(attr.GetGPUValues(), mask);
                    attr.Synchronize();
                }
                else if (relation1.types[i] == typeof(double))
                {
                    var attr = (_Attribute<double>)relation1.attributes[i];
                    this.ExecuteInnerJoinKernel<double>(attr.GetGPUValues(), mask);
                    attr.Synchronize();
                }
            }
        }
        */
    }
}
