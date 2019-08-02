using Microsoft.ML.Data;

namespace LinearRegression
{
    //  //make;model;price;currency;year;mileage;engine;fuel
    public class OtoMotoData
    {
        [ColumnName("make"), LoadColumn(0)]
        public string Make { get; set; }


        [ColumnName("model"), LoadColumn(1)]
        public string Model { get; set; }


        [ColumnName("price"), LoadColumn(2)]
        public float Price { get; set; }


        [ColumnName("year"), LoadColumn(3)]
        public float Year { get; set; }


        [ColumnName("mileage"), LoadColumn(4)]
        public float Mileage { get; set; }


        [ColumnName("engine"), LoadColumn(5)]
        public string Engine { get; set; }


        [ColumnName("fuel"), LoadColumn(6)]
        public string Fuel { get; set; }
    }

    public class PricePrediction
    {
        [ColumnName("Score")]
        public float price;
    }
}