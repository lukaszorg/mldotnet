using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using LinearRegressionML.Model.DataModels;
using Microsoft.ML;
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
        private ITransformer _model;

        public List<string> FuelTypes { get; } = new List<string>()
        {
            "Diesel",
            "Benzyna",
            "Benzyna+LPG",
        };

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
                _model = ModelBuilder.CreateModel(Columns, trainerOptions);
                return  ModelBuilder.GetMetrics(_model);
            });

            PrintRegressionMetrics(b);
            PredictPanel.Visibility = Visibility.Visible;
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

        private void PredictPriceClick(object sender, RoutedEventArgs e)
        {
            var predFunction = ModelBuilder.GetPredictFunction(_model);
            var carData = new CarModel
            {
                Make = tbMake.Text,
                Model = tbModel.Text,
                Year = float.Parse(tbYear.Text),
                Mileage = float.Parse(tbMileage.Text),
                Engine = float.Parse(tbEngine.Text),
                Fuel = (string) tbFuel.SelectedItem
            };

            var result =  predFunction.Predict(carData);

            tbPredictedPrice.Text = result.price.ToString("#") + " PLN";
        }
    }
}
