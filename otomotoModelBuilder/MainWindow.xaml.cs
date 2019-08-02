using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using OtomotoModelBuilder;

namespace otomotoModelBuilder
{
    public partial class MainWindow : Window
    {
        private readonly List<ColumnTransform> _stringNormalization = new List<ColumnTransform>() { ColumnTransform.OneHotEncoding, ColumnTransform.OneHotHashEncoding, ColumnTransform.None };
        private readonly List<ColumnTransform> _numberNormalization = new List<ColumnTransform>() { ColumnTransform.NormalizeMeanVariance, ColumnTransform.NormalizeMinMax, ColumnTransform.None };
        public ColumnModel[] Columns { get; set; }
        private LightGbmRegressionTrainerOptions trainerOptions = new LightGbmRegressionTrainerOptions();

        public MainWindow()
        {
            InitializeComponent();
            Initialize();

            DataContext = this;
            Editor.SelectedObject = trainerOptions;


        }

        private void Initialize()
        {
            Columns = new[]
            {
                new ColumnModel { Name = "make", SelectedTransform = ColumnTransform.OneHotEncoding, Transforms = _stringNormalization },
                new ColumnModel { Name = "model", SelectedTransform = ColumnTransform.OneHotEncoding, Transforms = _stringNormalization },
                new ColumnModel { Name = "year", SelectedTransform = ColumnTransform.NormalizeMinMax, Transforms = _numberNormalization },
                new ColumnModel { Name = "mileage", SelectedTransform = ColumnTransform.NormalizeMeanVariance, Transforms = _numberNormalization },
                new ColumnModel { Name = "engine", SelectedTransform = ColumnTransform.OneHotEncoding, Transforms = _stringNormalization },
                new ColumnModel { Name = "fuel", SelectedTransform = ColumnTransform.OneHotEncoding, Transforms = _stringNormalization },
            };
        }

        private async void Button_Click(object sender, RoutedEventArgs e)
        {

            GenerateButton.IsEnabled = false;
            ModelMetricsTextbox.Text = "Generating model..";
            var b = await Task.Run(() =>
            {
                var model = ModelBuilder.CreateModel(Columns, trainerOptions);
                return  ModelBuilder.GetMetrics(model);
            });

            PrintRegressionMetrics(b);
            GenerateButton.IsEnabled = true;

        }

        public void PrintRegressionMetrics(RegressionMetrics metrics)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"*************************************************");
            sb.AppendLine($"*       Metrics for regression model      ");
            sb.AppendLine($"*------------------------------------------------");
            sb.AppendLine($"*       LossFn:        {metrics.LossFunction:0.##}");
            sb.AppendLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            sb.AppendLine($"*       Absolute loss: {metrics.MeanAbsoluteError:#.##}");
            sb.AppendLine($"*       Squared loss:  {metrics.MeanSquaredError:#.##}");
            sb.AppendLine($"*       RMS loss:      {metrics.RootMeanSquaredError:#.##}");
            sb.AppendLine($"*************************************************");

            ModelMetricsTextbox.Text = sb.ToString();
        }

    }
}
