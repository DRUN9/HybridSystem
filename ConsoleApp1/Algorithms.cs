using ILGPU;
using ILGPU.Runtime;
using ConsoleTables;
using System.Diagnostics;
using System.Security.Cryptography;

class Program
{
    static void InnerJoinMaskKernel(Index1D index, ArrayView<int> array1, ArrayView<int> array2, ArrayView<int> result)
    {
        if (array1[index] == array2[index])
        {
            result[index] = index;
        }
        else
        {
            result[index] = -1;
        }
    }

    static void InnerJoinKernel(Index1D index, ArrayView<int> array, ArrayView<int> mask, ArrayView<int> result)     
    {
        if (mask[index] != -1)
        {
            result[index] = array[mask[index]];
        }
    }

    static void OuterJoinKernel1(Index1D index, ArrayView<int> array, ArrayView<int> result)
    {
        result[index] = array[index];
    }

    static void OuterJoinKernel2(Index1D index, ArrayView<int> array, ArrayView<int> mask, ArrayView<int> result)
    {
        if (mask[index] == 1)
        {
            result[index] = array[index];
        }
        else
        {
            result[index] = 0;
        }
    }

    static void OuterJoinMaskKernel(Index1D index, ArrayView<int> array1, ArrayView<int> array2, ArrayView<int> mask)
    {
        if (array1[index] == array2[index])
        {
            mask[index] = 1;
        }
        else
        {
            mask[index] = 0;
        }
    }

    static void IntersectionKernel(Index1D index, ArrayView<int> array, ArrayView<int> mask, ArrayView<int> result)
    {
        if (mask[index] == 1)
        {
            result[index] = array[index];
        }
        else
        {
            result[index] = 0;
        }
    }

    static void ExceptKernel(Index1D index, ArrayView<int> array, ArrayView<int> mask, ArrayView<int> result)
    {
        if (mask[index] == 0)
        {
            result[index] = array[index];
        }
        else
        {
            result[index] = 0;
        }
    }

    static void IntersectionMaskKernel(Index1D index, ArrayView<int> array1, ArrayView<int> array2, ArrayView<int> mask)
    {
        if (array1[index] != array2[index])
        {
            mask[index] = 0;
        }
    }

    static void CrossJoinKernel1(Index1D index, ArrayView<int> array, ArrayView<int> result)
    {
        int k = result.IntLength / array.IntLength;

        result[index] = array[index / k];
    }

    static void CrossJoinKernel2(Index1D index, ArrayView<int> array, ArrayView<int> result)
    {
        result[index] = array[index % array.IntLength];
    }

    static void UnionKernel(Index1D index, ArrayView<int> array1, ArrayView<int> array2, ArrayView<int> result)
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

    static void Union(int[] array1, int[] array2, int[] result)
    {
        Context context = Context.CreateDefault();
        Accelerator accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

        int length = array1.Length + array2.Length;
        var dArray1 = accelerator.Allocate1D<int>(array1.Length);
        var dArray2 = accelerator.Allocate1D<int>(array2.Length);
        var dResult = accelerator.Allocate1D<int>(length);

        dArray1.CopyFromCPU(array1);
        dArray2.CopyFromCPU(array2);

        var unionKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(UnionKernel);

        unionKernel(length, dArray1.View, dArray2.View, dResult.View);

        dResult.CopyToCPU(result);
    }

    static void CrossJoin(int[] array1, int[] array2, int[] result, bool flag)
    {
        Context context = Context.CreateDefault();
        Accelerator accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

        int length = array1.Length * array2.Length;
        var dArray1 = accelerator.Allocate1D<int>(array1.Length);
        var dArray2 = accelerator.Allocate1D<int>(array2.Length);
        var dResult = accelerator.Allocate1D<int>(length);

        dArray1.CopyFromCPU(array1);
        dArray2.CopyFromCPU(array2);

        var crossJoinKernel1 = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(CrossJoinKernel1);
        var crossJoinKernel2 = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(CrossJoinKernel2);

        Stopwatch stopwatch = new Stopwatch();

        stopwatch.Start();
        if (flag)
        {
            crossJoinKernel1(length, dArray1.View, dResult.View);
        }
        else
        {
            crossJoinKernel2(length, dArray2.View, dResult.View);
        }
        stopwatch.Stop();

        Console.WriteLine(stopwatch.ElapsedMilliseconds);

        dResult.CopyToCPU(result);
    }

    static void Except(int[] array1, int[] mask, int[] result)
    {
        Context context = Context.CreateDefault();
        Accelerator accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

        int length = array1.Length;
        var dArray1 = accelerator.Allocate1D<int>(length);
        var dResult = accelerator.Allocate1D<int>(length);
        var dMask = accelerator.Allocate1D<int>(length);

        dArray1.CopyFromCPU(array1);
        dMask.CopyFromCPU(mask);

        var intersectionKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(ExceptKernel);

        intersectionKernel(length, dArray1.View, dMask.View, dResult.View);

        dResult.CopyToCPU(result);
    }

    static void Intersection(int[] array1, int[] mask, int[] result)
    {
        Context context = Context.CreateDefault();
        Accelerator accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

        int length = array1.Length;
        var dArray1 = accelerator.Allocate1D<int>(length);
        var dResult = accelerator.Allocate1D<int>(length);
        var dMask = accelerator.Allocate1D<int>(length);

        dArray1.CopyFromCPU(array1);
        dMask.CopyFromCPU(mask);

        var intersectionKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(IntersectionKernel);

        intersectionKernel(length, dArray1.View, dMask.View, dResult.View);

        dResult.CopyToCPU(result);
    }

    static void IntersectionMask(int[] array1, int[] array2, int[] mask)
    {
        Context context = Context.CreateDefault();
        Accelerator accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

        int length = array1.Length;
        var dArray1 = accelerator.Allocate1D<int>(length);
        var dArray2 = accelerator.Allocate1D<int>(length);
        var dMask = accelerator.Allocate1D<int>(length);

        dArray1.CopyFromCPU(array1);
        dArray2.CopyFromCPU(array2);
        dMask.CopyFromCPU(mask);

        var intersectionMaskKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(IntersectionMaskKernel);
        intersectionMaskKernel(length, dArray1.View, dArray2.View, dMask.View);

        dMask.CopyToCPU(mask);
    }

    static void OuterJoinMask(int[] array1, int[] array2, int[] mask)
    {
        Context context = Context.CreateDefault();
        Accelerator accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

        int length = array1.Length;
        var dArray1 = accelerator.Allocate1D<int>(length);
        var dArray2 = accelerator.Allocate1D<int>(length);
        var dMask = accelerator.Allocate1D<int>(length);

        dArray1.CopyFromCPU(array1);
        dArray2.CopyFromCPU(array2);

        var outerJoinMaskKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(OuterJoinMaskKernel);
        outerJoinMaskKernel(length, dArray1.View, dArray2.View, dMask.View);

        dMask.CopyToCPU(mask);
    }

    static void LeftOuterJoin(int[] array, int[] mask, int[] result, bool flag)
    {
        Context context = Context.CreateDefault();
        Accelerator accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

        int length = array.Length;
        var dArray = accelerator.Allocate1D<int>(length);
        var dResult = accelerator.Allocate1D<int>(length);
        var dMask = accelerator.Allocate1D<int>(length);

        dArray.CopyFromCPU(array);
        dMask.CopyFromCPU(mask);

        var leftOuterJoinkernel1 = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(OuterJoinKernel1);
        var leftOuterJoinkernel2 = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(OuterJoinKernel2);

        if (flag)
        {
            leftOuterJoinkernel1(length, dArray.View, dResult.View);
        }
        else
        {
            leftOuterJoinkernel2(length, dArray.View, dMask.View, dResult.View);
        }

        dResult.CopyToCPU(result);
    }

    static void RightOuterJoin(int[] array, int[] mask, int[] result, bool flag)
    {
        Context context = Context.CreateDefault();
        Accelerator accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

        int length = array.Length;
        var dArray = accelerator.Allocate1D<int>(length);
        var dResult = accelerator.Allocate1D<int>(length);
        var dMask = accelerator.Allocate1D<int>(length);

        dArray.CopyFromCPU(array);
        dMask.CopyFromCPU(mask);

        var rightOuterJoinkernel1 = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(OuterJoinKernel1);
        var rightOuterJoinkernel2 = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(OuterJoinKernel2);

        if (flag)
        {
            rightOuterJoinkernel2(length, dArray.View, dMask.View, dResult.View);
        }
        else
        {
            rightOuterJoinkernel1(length, dArray.View, dResult.View);
        }

        dResult.CopyToCPU(result);
    }

    static void InnerJoinMask(int[] array1, int[] array2, int[] mask)
    {
        Context context = Context.CreateDefault();
        Accelerator accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

        int length = array1.Length;
        var dArray1 = accelerator.Allocate1D<int>(length);
        var dArray2 = accelerator.Allocate1D<int>(length);
        var dMask = accelerator.Allocate1D<int>(length);

        dArray1.CopyFromCPU(array1);
        dArray2.CopyFromCPU(array2);

        var innerJoinMaskKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(InnerJoinMaskKernel);
        innerJoinMaskKernel(length, dArray1.View, dArray2.View, dMask.View);

        dMask.CopyToCPU(mask);
    }

    static void InnerJoin(int[] array1, int[] mask, int[] result)
    {
        Context context = Context.CreateDefault();
        Accelerator accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

        int length = array1.Length;
        var dArray1 = accelerator.Allocate1D<int>(length);
        var dResult = accelerator.Allocate1D<int>(length);
        var dMask = accelerator.Allocate1D<int>(length);

        dArray1.CopyFromCPU(array1);
        dMask.CopyFromCPU(mask);

        var innerJoinkernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(InnerJoinKernel);
        innerJoinkernel(length, dArray1.View, dMask.View, dResult.View);

        dResult.CopyToCPU(result);
    }

    static int[] nonZeroCounter(int[] array)
    {
        int nonZeroCount = 0;
        foreach (var value in array)
        {
            if (value != 0)
            {
                nonZeroCount++;
            }
        }

        int[] finalResult = new int[nonZeroCount];
        int index = 0;

        foreach (var value in array)
        {
            if (value != 0)
            {
                finalResult[index++] = value;
            }
        }

        return finalResult;
    }

    static void InnerJoinDemo()
    {
        Random random = new Random();

        int[] h1 = { 1, 2, 3, 4, 5 };
        int[] h2 = { random.Next(1, 10), random.Next(1, 10), random.Next(1, 10), random.Next(1, 10), random.Next(1, 10) };
        int[] h3 = { 60000, 80000, 100000, 120000, 150000 };

        int[] h4 = { 1, 2, 3, 4, 5 };
        int[] h5 = { random.Next(1, 10), random.Next(1, 10), random.Next(1, 10), random.Next(1, 10), random.Next(1, 10) };
        int[] h6 = { 60000, 100, 100000, 31, 150000 };

        int[] mask = new int[5];

        int[] result1 = new int[5];
        int[] result2 = new int[5];
        int[] result3 = new int[5];
        int[] result4 = new int[5];
        int[] result5 = new int[5];
        int[] result6 = new int[5];

        InnerJoinMask(h3, h6, mask);

        Console.WriteLine("Результат фильтрации:");
        Console.WriteLine(string.Join(", ", mask));

        InnerJoin(h1, mask, result1);
        InnerJoin(h2, mask, result2);
        InnerJoin(h3, mask, result3);
        InnerJoin(h4, mask, result4);
        InnerJoin(h5, mask, result5);
        InnerJoin(h6, mask, result6);

        int[] finalResult1 = nonZeroCounter(result1);
        int[] finalResult2 = nonZeroCounter(result2);
        int[] finalResult3 = nonZeroCounter(result3);
        int[] finalResult4 = nonZeroCounter(result4);
        int[] finalResult5 = nonZeroCounter(result5);
        int[] finalResult6 = nonZeroCounter(result6);

        var table1 = new ConsoleTable("h1", "h2", "h3");
        for (int i = 0; i < h1.Length; i++)
        {
            table1.AddRow(h1[i], h2[i], h3[i]);
        }

        var table2 = new ConsoleTable("h4", "h5", "h6");
        for (int i = 0; i < h1.Length; i++)
        {
            table2.AddRow(h4[i], h5[i], h6[i]);
        }

        var table3 = new ConsoleTable("h1", "h2", "h3", "h4", "h5", "h6");
        for (int i = 0; i < finalResult1.Length; i++)
        {
            table3.AddRow(finalResult1[i], finalResult2[i], finalResult3[i], finalResult4[i], finalResult5[i], finalResult6[i]);
        }

        Console.WriteLine("Given matrix:");
        table1.Options.EnableCount = false;
        table1.Write();

        Console.WriteLine("Given matrix:");
        table2.Options.EnableCount = false;
        table2.Write();

        Console.WriteLine("Computations completed");
        Console.WriteLine("INNER JOIN Result:");
        table3.Options.EnableCount = false;
        table3.Write();
    }

    static void LeftOuterJoinDemo()
    {
        Random random = new Random();

        int[] h1 = { 1, 2, 3, 4, 5 };
        int[] h2 = { random.Next(1, 10), random.Next(1, 10), random.Next(1, 10), random.Next(1, 10), random.Next(1, 10) };
        int[] h3 = { 60000, 80000, 100000, 120000, 150000 };

        int[] h4 = { 1, 2, 3, 4, 5 };
        int[] h5 = { random.Next(1, 10), random.Next(1, 10), random.Next(1, 10), random.Next(1, 10), random.Next(1, 10) };
        int[] h6 = { 60000, 100, 100000, 31, 150000 };

        int[] mask = new int[5];

        int[] result1 = new int[5];
        int[] result2 = new int[5];
        int[] result3 = new int[5];
        int[] result4 = new int[5];
        int[] result5 = new int[5];
        int[] result6 = new int[5];

        OuterJoinMask(h3, h6, mask);

        LeftOuterJoin(h1, mask, result1, true);
        LeftOuterJoin(h2, mask, result2, true);
        LeftOuterJoin(h3, mask, result3, true);
        LeftOuterJoin(h4, mask, result4, false);
        LeftOuterJoin(h5, mask, result5, false);
        LeftOuterJoin(h6, mask, result6, false);

        var table1 = new ConsoleTable("h1", "h2", "h3");
        for (int i = 0; i < h1.Length; i++)
        {
            table1.AddRow(h1[i], h2[i], h3[i]);
        }

        var table2 = new ConsoleTable("h4", "h5", "h6");
        for (int i = 0; i < h1.Length; i++)
        {
            table2.AddRow(h4[i], h5[i], h6[i]);
        }

        var table3 = new ConsoleTable("h1", "h2", "h3", "h4", "h5", "h6");
        for (int i = 0; i < result1.Length; i++)
        {
            table3.AddRow(result1[i], result2[i], result3[i], result4[i], result5[i], result6[i]);
        }

        Console.WriteLine("Given matrix:");
        table1.Options.EnableCount = false;
        table1.Write();

        Console.WriteLine("Given matrix:");
        table2.Options.EnableCount = false;
        table2.Write();

        Console.WriteLine("Computations completed");
        Console.WriteLine("LEFT OUTER JOIN Result:");
        table3.Options.EnableCount = false;
        table3.Write();
    }

    static void RightOuterJoinDemo()
    {
        Random random = new Random();

        int[] h1 = { 1, 2, 3, 4, 5 };
        int[] h2 = { random.Next(1, 10), random.Next(1, 10), random.Next(1, 10), random.Next(1, 10), random.Next(1, 10) };
        int[] h3 = { 60000, 80000, 100000, 120000, 150000 };

        int[] h4 = { 1, 2, 3, 4, 5 };
        int[] h5 = { random.Next(1, 10), random.Next(1, 10), random.Next(1, 10), random.Next(1, 10), random.Next(1, 10) };
        int[] h6 = { 60000, 100, 100000, 31, 150000 };

        int[] mask = new int[5];

        int[] result1 = new int[5];
        int[] result2 = new int[5];
        int[] result3 = new int[5];
        int[] result4 = new int[5];
        int[] result5 = new int[5];
        int[] result6 = new int[5];

        OuterJoinMask(h3, h6, mask);

        RightOuterJoin(h1, mask, result1, true);
        RightOuterJoin(h2, mask, result2, true);
        RightOuterJoin(h3, mask, result3, true);
        RightOuterJoin(h4, mask, result4, false);
        RightOuterJoin(h5, mask, result5, false);
        RightOuterJoin(h6, mask, result6, false);

        var table1 = new ConsoleTable("h1", "h2", "h3");
        for (int i = 0; i < h1.Length; i++)
        {
            table1.AddRow(h1[i], h2[i], h3[i]);
        }

        var table2 = new ConsoleTable("h4", "h5", "h6");
        for (int i = 0; i < h1.Length; i++)
        {
            table2.AddRow(h4[i], h5[i], h6[i]);
        }

        var table3 = new ConsoleTable("h1", "h2", "h3", "h4", "h5", "h6");
        for (int i = 0; i < result1.Length; i++)
        {
            table3.AddRow(result1[i], result2[i], result3[i], result4[i], result5[i], result6[i]);
        }

        Console.WriteLine("Given matrix:");
        table1.Options.EnableCount = false;
        table1.Write();

        Console.WriteLine("Given matrix:");
        table2.Options.EnableCount = false;
        table2.Write();

        Console.WriteLine("Computations completed");
        Console.WriteLine("RIGHT OUTER JOIN Result:");
        table3.Options.EnableCount = false;
        table3.Write();
    }

    static void CrossJoinDemo()
    {
        Random random = new Random();

        int[] h1 = { 1, 2, 3 };
        int[] h2 = { random.Next(1, 10), random.Next(1, 10), random.Next(1, 10) };
        int[] h3 = { 60000, 80000, 100000 };

        int[] h4 = { 4, 5 };
        int[] h5 = { random.Next(1, 10), random.Next(1, 10) };
        int[] h6 = { 31, 100 };

        int[] result1 = new int[6];
        int[] result2 = new int[6];
        int[] result3 = new int[6];
        int[] result4 = new int[6];
        int[] result5 = new int[6];
        int[] result6 = new int[6];

        CrossJoin(h1, h4, result1, true);
        CrossJoin(h2, h4, result2, true);
        CrossJoin(h3, h4, result3, true);
        CrossJoin(h1, h4, result4, false);
        CrossJoin(h1, h5, result5, false);
        CrossJoin(h1, h6, result6, false);

        var table1 = new ConsoleTable("h1", "h2", "h3");
        for (int i = 0; i < h1.Length; i++)
        {
            table1.AddRow(h1[i], h2[i], h3[i]);
        }

        var table2 = new ConsoleTable("h4", "h5", "h6");
        for (int i = 0; i < h4.Length; i++)
        {
            table2.AddRow(h4[i], h5[i], h6[i]);
        }

        var table3 = new ConsoleTable("h1", "h2", "h3", "h4", "h5", "h6");
        for (int i = 0; i < result1.Length; i++)
        {
            table3.AddRow(result1[i], result2[i], result3[i], result4[i], result5[i], result6[i]);
        }

        Console.WriteLine("Given matrix:");
        table1.Options.EnableCount = false;
        table1.Write();

        Console.WriteLine("Given matrix:");
        table2.Options.EnableCount = false;
        table2.Write();

        Console.WriteLine("Computations completed");
        Console.WriteLine("CROSS JOIN Result:");
        table3.Options.EnableCount = false;
        table3.Write();
    }

    static void IntersectionDemo()
    {
        Random random = new Random();

        int[] h1 = { 1, 2, 3, 4, 5 };
        int[] h2 = { 2, 4, 6, 8, 10 };
        int[] h3 = { 60000, 80000, 100000, 120000, 150000 };

        int[] h4 = { 1, 2, 10, 4, 5 };
        int[] h5 = { 2, 7, 6, 8, 10 };
        int[] h6 = { 60000, 80000, 100000, 120000, 150000 };

        int[] mask = { 1, 1, 1, 1, 1 };

        int[] result1 = new int[5];
        int[] result2 = new int[5];
        int[] result3 = new int[5];

        IntersectionMask(h1, h4, mask);
        IntersectionMask(h2, h5, mask);
        IntersectionMask(h3, h6, mask);

        Intersection(h1, mask, result1);
        Intersection(h2, mask, result2);
        Intersection(h3, mask, result3);

        int[] finalResult1 = nonZeroCounter(result1);
        int[] finalResult2 = nonZeroCounter(result2);
        int[] finalResult3 = nonZeroCounter(result3);

        var table1 = new ConsoleTable("h1", "h2", "h3");
        for (int i = 0; i < h1.Length; i++)
        {
            table1.AddRow(h1[i], h2[i], h3[i]);
        }

        var table2 = new ConsoleTable("h1", "h2", "h3");
        for (int i = 0; i < h1.Length; i++)
        {
            table2.AddRow(h4[i], h5[i], h6[i]);
        }

        var table3 = new ConsoleTable("h1", "h2", "h3");
        for (int i = 0; i < finalResult1.Length; i++)
        {
            table3.AddRow(finalResult1[i], finalResult2[i], finalResult3[i]);
        }

        Console.WriteLine("Given matrix:");
        table1.Options.EnableCount = false;
        table1.Write();

        Console.WriteLine("Given matrix:");
        table2.Options.EnableCount = false;
        table2.Write();

        Console.WriteLine("Computations completed");
        Console.WriteLine("INTERSECT Result:");
        table3.Options.EnableCount = false;
        table3.Write();
    }

    static void UnionDemo()
    {
        Random random = new Random();

        int[] h1 = { 1, 2, 3, 4, 5 };
        int[] h2 = { 2, 4, 6, 8, 10 };
        int[] h3 = { 60000, 80000, 100000, 120000, 150000 };

        int[] h4 = { 1, 2, 10, 4, 5 };
        int[] h5 = { 2, 7, 6, 8, 10 };
        int[] h6 = { 60000, 80000, 100000, 120000, 150000 };

        int[] mask = { 1, 1, 1, 1, 1 };

        int[] result1 = new int[10];
        int[] result2 = new int[10];
        int[] result3 = new int[10];

        Union(h1, h4, result1);
        Union(h2, h5, result2);
        Union(h3, h6, result3);

        var table1 = new ConsoleTable("h1", "h2", "h3");
        for (int i = 0; i < h1.Length; i++)
        {
            table1.AddRow(h1[i], h2[i], h3[i]);
        }

        var table2 = new ConsoleTable("h1", "h2", "h3");
        for (int i = 0; i < h1.Length; i++)
        {
            table2.AddRow(h4[i], h5[i], h6[i]);
        }

        var table3 = new ConsoleTable("h1", "h2", "h3");
        for (int i = 0; i < result1.Length; i++)
        {
            table3.AddRow(result1[i], result2[i], result3[i]);
        }

        Console.WriteLine("Given matrix:");
        table1.Options.EnableCount = false;
        table1.Write();

        Console.WriteLine("Given matrix:");
        table2.Options.EnableCount = false;
        table2.Write();

        Console.WriteLine("Computations completed");
        Console.WriteLine("UNION Result:");
        table3.Options.EnableCount = false;
        table3.Write();
    }

    static void ExceptDemo()
    {
        Random random = new Random();

        int[] h1 = { 1, 2, 3, 4, 5 };
        int[] h2 = { 2, 4, 6, 8, 10 };
        int[] h3 = { 60000, 80000, 100000, 120000, 150000 };

        int[] h4 = { 1, 2, 10, 4, 5 };
        int[] h5 = { 2, 7, 6, 8, 10 };
        int[] h6 = { 60000, 80000, 100000, 120000, 150000 };

        int[] mask = { 1, 1, 1, 1, 1 };

        int[] result1 = new int[5];
        int[] result2 = new int[5];
        int[] result3 = new int[5];

        IntersectionMask(h1, h4, mask);
        IntersectionMask(h2, h5, mask);
        IntersectionMask(h3, h6, mask);

        Except(h1, mask, result1);
        Except(h2, mask, result2);
        Except(h3, mask, result3);

        int[] finalResult1 = nonZeroCounter(result1);
        int[] finalResult2 = nonZeroCounter(result2);
        int[] finalResult3 = nonZeroCounter(result3);

        var table1 = new ConsoleTable("h1", "h2", "h3");
        for (int i = 0; i < h1.Length; i++)
        {
            table1.AddRow(h1[i], h2[i], h3[i]);
        }

        var table2 = new ConsoleTable("h1", "h2", "h3");
        for (int i = 0; i < h1.Length; i++)
        {
            table2.AddRow(h4[i], h5[i], h6[i]);
        }

        var table3 = new ConsoleTable("h1", "h2", "h3");
        for (int i = 0; i < finalResult1.Length; i++)
        {
            table3.AddRow(finalResult1[i], finalResult2[i], finalResult3[i]);
        }

        Console.WriteLine("Given matrix:");
        table1.Options.EnableCount = false;
        table1.Write();

        Console.WriteLine("Given matrix:");
        table2.Options.EnableCount = false;
        table2.Write();

        Console.WriteLine("Computations completed");
        Console.WriteLine("EXCEPT Result:");
        table3.Options.EnableCount = false;
        table3.Write();
    }

    static int[] GenerateArray(int length)
    {
        Random random = new Random();
        int[] array = new int[length];
        for (int i = 0; i < length; i++)
        {
            array[i] = random.Next();
        }
        return array;
    }

    static void Main()
    {
        int bytes = 100;

        int elementSize = sizeof(int); // Размер одного int в байтах
        int arrayLength = bytes / elementSize;

        Console.Write("arrlen = ");
        Console.WriteLine(arrayLength);

        int[] a1 = GenerateArray(arrayLength);
        int[] a2 = GenerateArray(arrayLength);
        int[] res = new int[arrayLength * arrayLength];

        CrossJoin(a1, a2, res, true);

        //InnerJoinDemo();
        //LeftOuterJoinDemo();
        //RightOuterJoinDemo();
        //CrossJoinDemo();
        //IntersectionDemo();
        //ExceptDemo();
        //UnionDemo();
        
    }
}
