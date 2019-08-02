using System;
using Microsoft.ML.Data;

namespace LinearRegression
{
    public static class ConsoleHelper
    {
        public static void PrintRegressionMetrics(string name, RegressionMetrics metrics)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for {name} regression model      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       LossFn:        {metrics.LossFunction:0.##}");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Absolute loss: {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*       Squared loss:  {metrics.MeanAbsoluteError:#.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.MeanSquaredError:#.##}");
            Console.WriteLine($"*************************************************");
        }

        public static void ConsolePressAnyKey()
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine(" ");
            Console.WriteLine("Press any key to finish.");
            Console.ReadKey();
        }
    }
}
