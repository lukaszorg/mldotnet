namespace otomotoModelBuilder
{
    public class LightGbmRegressionTrainerOptions
    {
        public int NumberOfIterations { get; set; } = 50;
        public double LearningRate { get; set; } = 0.07721677f;
        public int NumberOfLeaves { get; set; } = 91;
        public int MinimumExampleCountPerLeaf { get; set; } = 20;
        public bool UseCategoricalSplit { get; set; } = true;
        public bool HandleMissingValue { get; set; } = true;
        public int MinimumExampleCountPerGroup { get; set; } = 100;
        public int MaximumCategoricalSplitPointCount { get; set; } = 8;
        public int CategoricalSmoothing { get; set; } = 20;
        public double L2CategoricalRegularization { get; set; } = 0.1;
        public double L2Regularization { get; set; } = 0;
        public double L1Regularization { get; set; } = 0.5;

    }
}