using System;
using System.Diagnostics;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using PLplot;

namespace LinearRegression
{
    class Program
    {
        static readonly string TrainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", @"otomoto.csv");
        static readonly string ModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);

            //STEP 0: Configure input data format
            //make;model;price;currency;year;mileage;engine;fuel
            var textLoader = mlContext.Data.CreateTextLoader(new[]
            {
                new TextLoader.Column("make", DataKind.String, 0),
                new TextLoader.Column("model", DataKind.String, 1),
                new TextLoader.Column("price", DataKind.Single, 2),
                new TextLoader.Column("year", DataKind.Single, 3),
                new TextLoader.Column("mileage", DataKind.Single, 4),
                new TextLoader.Column("engine", DataKind.String, 5),
                new TextLoader.Column("fuel", DataKind.String, 6)
            }, hasHeader: true, separatorChar: ',');

            //STEP 1: Load filter and shuffle training and test data
            var baseDataView = mlContext.Data.ShuffleRows(textLoader.Load(TrainDataPath));
            baseDataView = mlContext.Data.FilterRowsByColumn(baseDataView, "mileage", lowerBound: 10000, upperBound: 600000);
            baseDataView = mlContext.Data.FilterRowsByColumn(baseDataView, "year", lowerBound: 1980, upperBound: 2018);

            var trainTestData = mlContext.Data.TrainTestSplit(baseDataView, 0.001);
            var testDataView = trainTestData.TestSet;
            var trainingDataView = trainTestData.TrainSet;

            // STEP 2: Common data process configuration with pipeline data transformations
            var dataProcessPipeline = mlContext.Transforms.Categorical.OneHotEncoding(new[]
                {
                    new InputOutputColumnPair("engine"),
                    new InputOutputColumnPair("make"),
                    new InputOutputColumnPair("fuel")
                })
                .Append(mlContext.Transforms.NormalizeMeanVariance("mileage"))
                .Append(mlContext.Transforms.NormalizeMeanVariance("year"))
                .Append(mlContext.Transforms.Categorical.OneHotHashEncoding(new[]{new InputOutputColumnPair("model")}))
                .Append(mlContext.Transforms.Concatenate("Features",new[] {"make", "fuel", "model", "year", "mileage", "engine"}));

            // STEP 3: Set the training algorithm 
            var trainer = mlContext.Regression.Trainers.LightGbm(new LightGbmRegressionTrainer.Options()
            {
                NumberOfIterations = 50,
                LearningRate = 0.07721677f,
                NumberOfLeaves = 91,
                MinimumExampleCountPerLeaf = 20,
                UseCategoricalSplit = true,
                HandleMissingValue = true,
                MinimumExampleCountPerGroup = 100,
                MaximumCategoricalSplitPointCount = 8,
                CategoricalSmoothing = 20,
                L2CategoricalRegularization = 0.1,
                Booster = new GradientBooster.Options()
                {
                    L2Regularization = 0,
                    L1Regularization = 0.5
                },
                LabelColumnName = "price",
                FeatureColumnName = "Features"
            });
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            Console.WriteLine("=============== Training the model ===============");
            //STEP 4: Train the model
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            //STEP 5: Calculate model metrics
            var predictions = trainedModel.Transform(trainingDataView);
            var metrics =
                mlContext.Regression.Evaluate(predictions, labelColumnName: "price", scoreColumnName: "Score");
            ConsoleHelper.PrintRegressionMetrics(trainer.ToString(), metrics);

            //STEP 6: Save generated model to file
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);
            Console.WriteLine("The model is saved to {0}", ModelPath);

            //STEP 7: 
            PlotRegressionChart(mlContext, trainedModel, testDataView, 30, new[] {"svg"});

            //return trainedModel;
            ConsoleHelper.ConsolePressAnyKey();
        }

        private static void PlotRegressionChart(MLContext mlContext, ITransformer trainedModel, IDataView testData,
            int numberOfRecordsToRead, string[] args)
        {
            // Create prediction engine related to the loaded trained model
            var predFunction =
                mlContext.Model.CreatePredictionEngine<OtoMotoData, PricePrediction>(trainedModel);

            string chartFileName = "";

            using (var pl = new PLStream())
            {
                if (args.Length == 1 && args[0] == "svg")
                {
                    pl.sdev("svg");
                    chartFileName = "RegressionDistribution.svg";
                    pl.sfnam(chartFileName);
                }
                else
                {
                    pl.sdev("pngcairo");
                    chartFileName = "RegressionDistribution.png";
                    pl.sfnam(chartFileName);
                }

                // use white background with black foreground
                pl.spal0("cmap0_alternate.pal");

                // Initialize plplot
                pl.init();

                // set axis limits
                const int xMinLimit = 0;
                const int xMaxLimit = 60000; //Rides larger than $35 are not shown in the chart
                const int yMinLimit = 0;
                const int yMaxLimit = 60000; //Rides larger than $35 are not shown in the chart
                pl.env(xMinLimit, xMaxLimit, yMinLimit, yMaxLimit, AxesScale.Independent, AxisBox.BoxTicksLabelsAxes);

                // Set scaling for mail title text 125% size of default
                pl.schr(0, 1.25);

                // The main title
                pl.lab("Measured", "Predicted", "Distribution of price prediction");

                // plot using different colors
                // see http://plplot.sourceforge.net/examples.php?demo=02 for palette indices
                pl.col0(1);

                int totalNumber = numberOfRecordsToRead;

                //This code is the symbol to paint
                char code = (char) 9;

                // plot using other color
                pl.col0(2); //Blue

                double yTotal = 0;
                double xTotal = 0;
                double xyMultiTotal = 0;
                double xSquareTotal = 0;

                foreach (var td in mlContext.Data.CreateEnumerable<OtoMotoData>(testData, reuseRowObject: true))
                {
                    var x = new double[1];
                    var y = new double[1];

                    //Make Prediction
                    var prediction = predFunction.Predict(td);

                    x[0] = td.Price;
                    y[0] = prediction.price;


                    //Paint a dot
                    pl.poin(x, y, code);

                    xTotal += x[0];
                    yTotal += y[0];

                    double multi = x[0] * y[0];
                    xyMultiTotal += multi;

                    double xSquare = x[0] * x[0];
                    xSquareTotal += xSquare;

                    double ySquare = y[0] * y[0];
                }

                // Regression Line calculation explanation:
                // https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data/more-on-regression/v/regression-line-example

                double minY = yTotal / totalNumber;
                double minX = xTotal / totalNumber;
                double minXY = xyMultiTotal / totalNumber;
                double minXsquare = xSquareTotal / totalNumber;

                double m = Math.Abs(((minX * minY) - minXY) / ((minX * minX) - minXsquare));

                double b = minY - (m * minX);

                //Generic function for Y for the regression line
                // y = (m * x) + b;

                double x1 = 1;
                //Function for Y1 in the line
                double y1 = 0; //(m * x1) + b;

                double x2 = 50000;
                //Function for Y2 in the line
                double y2 = (m * x2) + b;

                var xArray = new double[2];
                var yArray = new double[2];
                xArray[0] = x1;
                yArray[0] = y1;
                xArray[1] = x2;
                yArray[1] = y2;

                pl.col0(4);
                pl.line(xArray, yArray);

                // end page (writes output to disk)
                pl.eop();

                // output version of PLplot
                pl.gver(out var verText);
                Console.WriteLine("PLplot version " + verText);

            } // the pl object is disposed here

            Console.WriteLine("Showing chart...");
            var p = new Process();
            string chartFileNamePath = @".\" + chartFileName;
            p.StartInfo = new ProcessStartInfo(chartFileNamePath)
            {
                UseShellExecute = true
            };
            p.Start();
        }
    }
}