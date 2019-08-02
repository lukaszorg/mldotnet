using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using LinearRegressionML.Model.DataModels;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms;
using OtomotoModelBuilder;

namespace otomotoModelBuilder
{
    public static class ModelBuilder
    {
        private static readonly string TrainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", @"otomoto.csv");

        // Create MLContext to be shared across the model creation workflow objects 
        // Set a random seed for repeatable/deterministic results across multiple trainings.
        private static MLContext mlContext = new MLContext(seed: 1);

        public static ITransformer CreateModel(ColumnModel[] colums, LightGbmRegressionTrainerOptions trainerOptions)
        {
            // Load Data
            IDataView dataView = mlContext.Data.LoadFromTextFile<CarModel>(
                                            path: TrainDataPath,
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);
            dataView = mlContext.Data.ShuffleRows(dataView);
            var data = mlContext.Data.TrainTestSplit(dataView, 0.2);
            var trainingDataView = data.TrainSet;

            // Build training pipeline
            IEstimator<ITransformer> trainingPipeline = BuildTrainingPipeline(mlContext, colums, trainerOptions);


           // Evaluate(mlContext, data.TestSet, trainingPipeline);
            // Train Model
            ITransformer model = trainingPipeline.Fit(trainingDataView);

            return model;
        }

        public static RegressionMetrics GetMetrics(ITransformer model)
        {
            // Load Data
            IDataView dataView = mlContext.Data.LoadFromTextFile<CarModel>(
                path: TrainDataPath,
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true,
                allowSparse: false);
            dataView = mlContext.Data.ShuffleRows(dataView);
            var data = mlContext.Data.TrainTestSplit(dataView, 0.2);
            
            IDataView predictions = model.Transform(data.TestSet);
            return mlContext.Regression.Evaluate(predictions, labelColumnName: "price", scoreColumnName: "Score");
        }

        public static IEstimator<ITransformer> BuildTrainingPipeline(MLContext mlContext, ColumnModel[] colums,
            LightGbmRegressionTrainerOptions trainerOptions)
        {
            IEstimator<ITransformer> dataProcessPipeline = mlContext.Transforms.CustomMapping<object, object>((i, i1) => { }, "empty");

            foreach (var column in colums)
            {
                var transform = column.SelectedTransform;
                if(transform == ColumnTransform.OneHotEncoding)
                {
                    dataProcessPipeline = dataProcessPipeline.Append(mlContext.Transforms.Categorical.OneHotEncoding(column.Name));
                }
                else if(transform == ColumnTransform.OneHotHashEncoding)
                {
                    dataProcessPipeline = dataProcessPipeline.Append(mlContext.Transforms.Categorical.OneHotHashEncoding(column.Name));
                }
                else if (transform == ColumnTransform.NormalizeMeanVariance)
                {

                    dataProcessPipeline =  dataProcessPipeline.Append(mlContext.Transforms.NormalizeMeanVariance(column.Name));
                }
                else if (transform == ColumnTransform.NormalizeMinMax)
                {

                    dataProcessPipeline =  dataProcessPipeline.Append(mlContext.Transforms.NormalizeMinMax(column.Name));
                }
            }

            dataProcessPipeline = dataProcessPipeline.Append(mlContext.Transforms.Concatenate("Features", colums.Select(c=>c.Name).ToArray()));
            // dataProcessPipeline = dataProcessPipeline.Append(mlContext.Transforms.NormalizeMinMax("Features"));

            // Set the training algorithm 

            var trainer = mlContext.Regression.Trainers.LightGbm(new LightGbmRegressionTrainer.Options()
            {
                NumberOfIterations = trainerOptions.NumberOfIterations,
                LearningRate = trainerOptions.LearningRate,
                NumberOfLeaves = trainerOptions.NumberOfLeaves,
                MinimumExampleCountPerLeaf = trainerOptions.MinimumExampleCountPerLeaf,
                UseCategoricalSplit = trainerOptions.UseCategoricalSplit,
                HandleMissingValue = trainerOptions.HandleMissingValue,
                MinimumExampleCountPerGroup = trainerOptions.MinimumExampleCountPerGroup,
                MaximumCategoricalSplitPointCount = trainerOptions.MaximumCategoricalSplitPointCount,
                CategoricalSmoothing = trainerOptions.CategoricalSmoothing,
                L2CategoricalRegularization = trainerOptions.L2CategoricalRegularization,
                Booster = new GradientBooster.Options() {
                        L2Regularization = trainerOptions.L2Regularization,
                        L1Regularization = trainerOptions.L1Regularization },
                LabelColumnName = "price",
                FeatureColumnName = "Features" });

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            return trainingPipeline;
        }


        private static void Evaluate(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
            Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults = mlContext.Regression.CrossValidate(trainingDataView, trainingPipeline, numberOfFolds: 5, labelColumnName: "price");
            PrintRegressionFoldsAverageMetrics(crossValidationResults);
        }

        private static void SaveModel(MLContext mlContext, ITransformer mlModel, string modelRelativePath, DataViewSchema modelInputSchema)
        {
            // Save/persist the trained model to a .ZIP file
            Console.WriteLine($"=============== Saving the model  ===============");
            mlContext.Model.Save(mlModel, modelInputSchema, GetAbsolutePath(modelRelativePath));
            Console.WriteLine("The model is saved to {0}", GetAbsolutePath(modelRelativePath));
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(ModelBuilder).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }


        public static void PrintRegressionFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<RegressionMetrics>> crossValidationResults)
        {
            var L1 = crossValidationResults.Select(r => r.Metrics.MeanAbsoluteError);
            var L2 = crossValidationResults.Select(r => r.Metrics.MeanSquaredError);
            var RMS = crossValidationResults.Select(r => r.Metrics.RootMeanSquaredError);
            var lossFunction = crossValidationResults.Select(r => r.Metrics.LossFunction);
            var R2 = crossValidationResults.Select(r => r.Metrics.RSquared);

            Debug.WriteLine($"*************************************************************************************************************");
            Debug.WriteLine($"*       Metrics for Regression model      ");
            Debug.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Debug.WriteLine($"*       Average L1 Loss:       {L1.Average():0.###} ");
            Debug.WriteLine($"*       Average L2 Loss:       {L2.Average():0.###}  ");
            Debug.WriteLine($"*       Average RMS:           {RMS.Average():0.###}  ");
            Debug.WriteLine($"*       Average Loss Function: {lossFunction.Average():0.###}  ");
            Debug.WriteLine($"*       Average R-squared:     {R2.Average():0.###}  ");
            Debug.WriteLine($"*************************************************************************************************************");
        }

        public static PredictionEngine<CarModel, PricePrediction> GetPredictFunction(ITransformer model)
        {
            return mlContext.Model.CreatePredictionEngine<CarModel, PricePrediction>(model);
        }
    }

    public class PricePrediction
    {
        [ColumnName("Score")]
        public float price;
    }
}
