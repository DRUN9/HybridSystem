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
        public Dictionary<Type, Delegate> crossJoinKernels1;
        public Dictionary<Type, Delegate> crossJoinKernels2;
        public Dictionary<Type, Delegate> unionKernels;

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
            this.crossJoinKernels1 = new Dictionary<Type, Delegate>();
            this.crossJoinKernels2 = new Dictionary<Type, Delegate>();
            this.unionKernels = new Dictionary<Type, Delegate>();

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

            Action<Index1D, ArrayView1D<Optional<int>, Stride1D.Dense>, ArrayView1D<Optional<int>, Stride1D.Dense>> loadedCrossJoinKernel1Int = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Optional<int>, Stride1D.Dense>, ArrayView1D<Optional<int>, Stride1D.Dense>>(CrossJoinKernel1);
            Action<Index1D, ArrayView1D<Optional<float>, Stride1D.Dense>, ArrayView1D<Optional<float>, Stride1D.Dense>> loadedCrossJoinKernel1Float = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Optional<float>, Stride1D.Dense>, ArrayView1D<Optional<float>, Stride1D.Dense>>(CrossJoinKernel1);
            Action<Index1D, ArrayView1D<Optional<double>, Stride1D.Dense>, ArrayView1D<Optional<double>, Stride1D.Dense>> loadedCrossJoinKernel1Double = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Optional<double>, Stride1D.Dense>, ArrayView1D<Optional<double>, Stride1D.Dense>>(CrossJoinKernel1);

            this.crossJoinKernels1[typeof(int)] = loadedCrossJoinKernel1Int;
            this.crossJoinKernels1[typeof(float)] = loadedCrossJoinKernel1Float;
            this.crossJoinKernels1[typeof(double)] = loadedCrossJoinKernel1Double;

            Action<Index1D, ArrayView1D<Optional<int>, Stride1D.Dense>, ArrayView1D<Optional<int>, Stride1D.Dense>> loadedCrossJoinKernel2Int = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Optional<int>, Stride1D.Dense>, ArrayView1D<Optional<int>, Stride1D.Dense>>(CrossJoinKernel2);
            Action<Index1D, ArrayView1D<Optional<float>, Stride1D.Dense>, ArrayView1D<Optional<float>, Stride1D.Dense>> loadedCrossJoinKernel2Float = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Optional<float>, Stride1D.Dense>, ArrayView1D<Optional<float>, Stride1D.Dense>>(CrossJoinKernel2);
            Action<Index1D, ArrayView1D<Optional<double>, Stride1D.Dense>, ArrayView1D<Optional<double>, Stride1D.Dense>> loadedCrossJoinKernel2Double = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Optional<double>, Stride1D.Dense>, ArrayView1D<Optional<double>, Stride1D.Dense>>(CrossJoinKernel2);

            this.crossJoinKernels2[typeof(int)] = loadedCrossJoinKernel2Int;
            this.crossJoinKernels2[typeof(float)] = loadedCrossJoinKernel2Float;
            this.crossJoinKernels2[typeof(double)] = loadedCrossJoinKernel2Double;

            Action<Index1D, ArrayView1D<Optional<int>, Stride1D.Dense>, ArrayView1D<Optional<int>, Stride1D.Dense>, ArrayView1D<Optional<int>, Stride1D.Dense>> loadedUnionKernelInt = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Optional<int>, Stride1D.Dense>, ArrayView1D<Optional<int>, Stride1D.Dense>, ArrayView1D<Optional<int>, Stride1D.Dense>>(UnionKernel);
            Action<Index1D, ArrayView1D<Optional<float>, Stride1D.Dense>, ArrayView1D<Optional<float>, Stride1D.Dense>, ArrayView1D<Optional<float>, Stride1D.Dense>> loadedUnionKernelFloat = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Optional<float>, Stride1D.Dense>, ArrayView1D<Optional<float>, Stride1D.Dense>, ArrayView1D<Optional<float>, Stride1D.Dense>>(UnionKernel);
            Action<Index1D, ArrayView1D<Optional<double>, Stride1D.Dense>, ArrayView1D<Optional<double>, Stride1D.Dense>, ArrayView1D<Optional<double>, Stride1D.Dense>> loadedUnionKernelDouble = this.accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Optional<double>, Stride1D.Dense>, ArrayView1D<Optional<double>, Stride1D.Dense>, ArrayView1D<Optional<double>, Stride1D.Dense>>(UnionKernel);

            this.unionKernels[typeof(int)] = loadedUnionKernelInt;
            this.unionKernels[typeof(float)] = loadedUnionKernelFloat;
            this.unionKernels[typeof(double)] = loadedUnionKernelDouble;

        }

        static void UnionKernel<T>(Index1D index, ArrayView1D<T, Stride1D.Dense> array1, ArrayView1D<T, Stride1D.Dense> array2, ArrayView1D<T, Stride1D.Dense> result) where T : unmanaged
        {
            if (index < array1.IntLength)
            {
                result[index] = array1[index];
            }
            else
            {
                result[index] = array2[index - array1.IntLength];
            }
        }

        static void CrossJoinKernel1<T>(Index1D index, ArrayView1D<T, Stride1D.Dense> array, ArrayView1D<T, Stride1D.Dense> result) where T : unmanaged
        {
            long k = result.IntLength / array.IntLength;

            result[index] = array[index / k];
        }

        static void CrossJoinKernel2<T>(Index1D index, ArrayView1D<T, Stride1D.Dense> array, ArrayView1D<T, Stride1D.Dense> result) where T : unmanaged
        {
            result[index] = array[index % array.Length];
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

        void ExecuteCrossJoin<T>(MemoryBuffer1D<T, Stride1D.Dense> array, MemoryBuffer1D<T, Stride1D.Dense> result, bool flag, int length) where T : unmanaged
        {
            var kernelDelegate1 = (Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>>)this.crossJoinKernels1[typeof(T)];
            var kernelDelegate2 = (Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>>)this.crossJoinKernels2[typeof(T)];

            this.accelerator.Synchronize();
            if (flag)
            {
                kernelDelegate1(length, array.View, result.View);
            }
            else
            {
                kernelDelegate2(length, array.View, result.View);
            }
            this.accelerator.Synchronize();
        }

        void ExecuteUnion<T>(MemoryBuffer1D<T, Stride1D.Dense> array1, MemoryBuffer1D<T, Stride1D.Dense> array2, MemoryBuffer1D<T, Stride1D.Dense> result) where T : unmanaged
        {
            var kernelDelegate = (Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>>)this.unionKernels[typeof(T)];

            this.accelerator.Synchronize();
            kernelDelegate((Index1D)result.Length, array1.View, array2.View, result.View);
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

        public Relation CrossJoin(Relation relation1, Relation relation2)
        {
            Relation result = new Relation(relation1.name + " CROSS JOIN " + relation2.name, relation1.attributeNames.Concat(relation2.attributeNames).ToArray());

            for (int i = 0; i < relation1.Count + relation2.Count; ++i)
            {
                if (i < relation1.Count)
                {
                    if (relation1.types[i] == typeof(Optional<int>))
                    {
                        Optional<int>[] new_data = new Optional<int>[relation1.attributes[0].Count * relation2.attributes[0].Count];
                        _Attribute<Optional<int>> new_attr = new _Attribute<Optional<int>>(relation1.attributeNames[i], this.accelerator, new_data);
                        result.attributes.Add(new_attr);
                    }
                    else if (relation1.types[i] == typeof(Optional<float>))
                    {
                        Optional<float>[] new_data = new Optional<float>[relation1.attributes[0].Count * relation2.attributes[0].Count];
                        _Attribute<Optional<float>> new_attr = new _Attribute<Optional<float>>(relation1.attributeNames[i], this.accelerator, new_data);
                        result.attributes.Add(new_attr);
                    }
                    else if (relation1.types[i] == typeof(Optional<double>))
                    {
                        Optional<double>[] new_data = new Optional<double>[relation1.attributes[0].Count * relation2.attributes[0].Count];
                        _Attribute<Optional<double>> new_attr = new _Attribute<Optional<double>>(relation1.attributeNames[i], this.accelerator, new_data);
                        result.attributes.Add(new_attr);
                    }
                }
                else
                {
                    if (relation2.types[i - relation1.Count] == typeof(Optional<int>))
                    {
                        Optional<int>[] new_data = new Optional<int>[relation1.attributes[0].Count * relation2.attributes[0].Count];
                        _Attribute<Optional<int>> new_attr = new _Attribute<Optional<int>>(relation2.attributeNames[i - relation1.Count], this.accelerator, new_data);
                        result.attributes.Add(new_attr);
                    }
                    else if (relation2.types[i - relation1.Count] == typeof(Optional<float>))
                    {
                        Optional<float>[] new_data = new Optional<float>[relation1.attributes[0].Count * relation2.attributes[0].Count];
                        _Attribute<Optional<float>> new_attr = new _Attribute<Optional<float>>(relation2.attributeNames[i - relation1.Count], this.accelerator, new_data);
                        result.attributes.Add(new_attr);
                    }
                    else if (relation2.types[i - relation1.Count] == typeof(Optional<double>))
                    {
                        Optional<double>[] new_data = new Optional<double>[relation1.attributes[0].Count * relation2.attributes[0].Count];
                        _Attribute<Optional<double>> new_attr = new _Attribute<Optional<double>>(relation2.attributeNames[i - relation2.Count], this.accelerator, new_data);
                        result.attributes.Add(new_attr);
                    }
                }
            }

            for (int i = 0; i < relation1.Count; ++i)
            {
                if (relation1.types[i] == typeof(Optional<int>))
                {
                    var attr = (_Attribute<Optional<int>>)relation1.attributes[i];
                    var res_attr = (_Attribute<Optional<int>>)result.attributes[i];

                    this.ExecuteCrossJoin<Optional<int>>(attr.GetGPUValues(), res_attr.GetGPUValues(), true, res_attr.Count);
                    res_attr.Synchronize();
                }
                else if (relation1.types[i] == typeof(Optional<float>))
                {
                    var attr = (_Attribute<Optional<float>>)relation1.attributes[i];
                    var res_attr = (_Attribute<Optional<float>>)result.attributes[i];

                    this.ExecuteCrossJoin<Optional<float>>(attr.GetGPUValues(), res_attr.GetGPUValues(), true, res_attr.Count);
                    res_attr.Synchronize();
                }
                else if (relation1.types[i] == typeof(Optional<double>))
                {
                    var attr = (_Attribute<Optional<double>>)relation1.attributes[i];
                    var res_attr = (_Attribute<Optional<double>>)result.attributes[i];

                    this.ExecuteCrossJoin<Optional<double>>(attr.GetGPUValues(), res_attr.GetGPUValues(), true, res_attr.Count);
                    res_attr.Synchronize();
                }
            }

            for (int i = 0; i < relation2.Count; ++i)
            {
                if (relation2.types[i] == typeof(Optional<int>))
                {
                    var attr = (_Attribute<Optional<int>>)relation1.attributes[i];
                    var res_attr = (_Attribute<Optional<int>>)result.attributes[i + relation1.Count];

                    this.ExecuteCrossJoin<Optional<int>>(attr.GetGPUValues(), res_attr.GetGPUValues(), false, res_attr.Count);
                    res_attr.Synchronize();
                }
                else if (relation2.types[i] == typeof(Optional<float>))
                {
                    var attr = (_Attribute<Optional<float>>)relation1.attributes[i];
                    var res_attr = (_Attribute<Optional<float>>)result.attributes[i + relation2.Count];

                    this.ExecuteCrossJoin<Optional<float>>(attr.GetGPUValues(), res_attr.GetGPUValues(), false, res_attr.Count);
                    res_attr.Synchronize();
                }
                else if (relation2.types[i] == typeof(Optional<double>))
                {
                    var attr = (_Attribute<Optional<double>>)relation1.attributes[i];
                    var res_attr = (_Attribute<Optional<double>>)result.attributes[i + relation2.Count];

                    this.ExecuteCrossJoin<Optional<double>>(attr.GetGPUValues(), res_attr.GetGPUValues(), false, res_attr.Count);
                    res_attr.Synchronize();
                }
            }

            return result;
        }

        public Relation Union(Relation relation1, Relation relation2)
        {
            Relation result = new Relation(relation1.name + " UNION " + relation2.name, relation1.attributeNames);

            for (int i = 0; i < relation1.Count + relation2.Count; ++i)
            {
                if (i < relation1.Count)
                {
                    if (relation1.types[i] == typeof(Optional<int>))
                    {
                        Optional<int>[] new_data = new Optional<int>[relation1.attributes[0].Count + relation2.attributes[0].Count];
                        _Attribute<Optional<int>> new_attr = new _Attribute<Optional<int>>(relation1.attributeNames[i], this.accelerator, new_data);
                        result.attributes.Add(new_attr);
                    }
                    else if (relation1.types[i] == typeof(Optional<float>))
                    {
                        Optional<float>[] new_data = new Optional<float>[relation1.attributes[0].Count + relation2.attributes[0].Count];
                        _Attribute<Optional<float>> new_attr = new _Attribute<Optional<float>>(relation1.attributeNames[i], this.accelerator, new_data);
                        result.attributes.Add(new_attr);
                    }
                    else if (relation1.types[i] == typeof(Optional<double>))
                    {
                        Optional<double>[] new_data = new Optional<double>[relation1.attributes[0].Count + relation2.attributes[0].Count];
                        _Attribute<Optional<double>> new_attr = new _Attribute<Optional<double>>(relation1.attributeNames[i], this.accelerator, new_data);
                        result.attributes.Add(new_attr);
                    }
                }
                else
                {
                    if (relation2.types[i - relation1.Count] == typeof(Optional<int>))
                    {
                        Optional<int>[] new_data = new Optional<int>[relation1.attributes[0].Count * relation2.attributes[0].Count];
                        _Attribute<Optional<int>> new_attr = new _Attribute<Optional<int>>(relation2.attributeNames[i - relation1.Count], this.accelerator, new_data);
                        result.attributes.Add(new_attr);
                    }
                    else if (relation2.types[i - relation1.Count] == typeof(Optional<float>))
                    {
                        Optional<float>[] new_data = new Optional<float>[relation1.attributes[0].Count * relation2.attributes[0].Count];
                        _Attribute<Optional<float>> new_attr = new _Attribute<Optional<float>>(relation2.attributeNames[i - relation1.Count], this.accelerator, new_data);
                        result.attributes.Add(new_attr);
                    }
                    else if (relation2.types[i - relation1.Count] == typeof(Optional<double>))
                    {
                        Optional<double>[] new_data = new Optional<double>[relation1.attributes[0].Count * relation2.attributes[0].Count];
                        _Attribute<Optional<double>> new_attr = new _Attribute<Optional<double>>(relation2.attributeNames[i - relation2.Count], this.accelerator, new_data);
                        result.attributes.Add(new_attr);
                    }
                }
            }

            for (int i = 0; i < relation1.Count; ++i)
            {
                if (relation1.types[i] == typeof(Optional<int>))
                {
                    var attr = (_Attribute<Optional<int>>)relation1.attributes[i];
                    var res_attr = (_Attribute<Optional<int>>)result.attributes[i];

                    this.ExecuteCrossJoin<Optional<int>>(attr.GetGPUValues(), res_attr.GetGPUValues(), true, res_attr.Count);
                    res_attr.Synchronize();
                }
                else if (relation1.types[i] == typeof(Optional<float>))
                {
                    var attr = (_Attribute<Optional<float>>)relation1.attributes[i];
                    var res_attr = (_Attribute<Optional<float>>)result.attributes[i];

                    this.ExecuteCrossJoin<Optional<float>>(attr.GetGPUValues(), res_attr.GetGPUValues(), true, res_attr.Count);
                    res_attr.Synchronize();
                }
                else if (relation1.types[i] == typeof(Optional<double>))
                {
                    var attr = (_Attribute<Optional<double>>)relation1.attributes[i];
                    var res_attr = (_Attribute<Optional<double>>)result.attributes[i];

                    this.ExecuteCrossJoin<Optional<double>>(attr.GetGPUValues(), res_attr.GetGPUValues(), true, res_attr.Count);
                    res_attr.Synchronize();
                }
            }

            for (int i = 0; i < relation2.Count; ++i)
            {
                if (relation2.types[i] == typeof(Optional<int>))
                {
                    var attr = (_Attribute<Optional<int>>)relation1.attributes[i];
                    var res_attr = (_Attribute<Optional<int>>)result.attributes[i + relation1.Count];

                    this.ExecuteCrossJoin<Optional<int>>(attr.GetGPUValues(), res_attr.GetGPUValues(), true, res_attr.Count);
                    res_attr.Synchronize();
                }
                else if (relation2.types[i] == typeof(Optional<float>))
                {
                    var attr = (_Attribute<Optional<float>>)relation1.attributes[i];
                    var res_attr = (_Attribute<Optional<float>>)result.attributes[i + relation2.Count];

                    this.ExecuteCrossJoin<Optional<float>>(attr.GetGPUValues(), res_attr.GetGPUValues(), true, res_attr.Count);
                    res_attr.Synchronize();
                }
                else if (relation2.types[i] == typeof(Optional<double>))
                {
                    var attr = (_Attribute<Optional<double>>)relation1.attributes[i];
                    var res_attr = (_Attribute<Optional<double>>)result.attributes[i + relation2.Count];

                    this.ExecuteCrossJoin<Optional<double>>(attr.GetGPUValues(), res_attr.GetGPUValues(), true, res_attr.Count);
                    res_attr.Synchronize();
                }
            }

            return result;
        }
    }
}
