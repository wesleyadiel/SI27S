using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using StockForecast.Models;
using static Microsoft.ML.DataOperationsCatalog;
using static Microsoft.ML.Transforms.MissingValueReplacingEstimator;

namespace StockForecast
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext context = new MLContext();

            TrainTestData splitData = Sanitize(context);
            ITransformer model = Train(context, splitData.TrainSet);
            RegressionMetrics metrics = Evaluate(context, model, splitData.TestSet);

            VerificarMetricas(metrics);
            RealizarPrevisao(context, model);

            Console.ReadLine();
        }

        private static TrainTestData Sanitize(MLContext context)
        {
            IDataView dataview = context.Data.LoadFromTextFile<BTCCSV>("../Data/BTC-USD.csv", ',', true);
            dataview = context.Data.FilterRowsByMissingValues(dataview, "Abertura", "Maxima", "Minima", "FechamentoAjustado", "Volume");
            TrainTestData trainTestData = context.Data.TrainTestSplit(dataview, 0.25);

            return trainTestData;
        }

        private static ITransformer Train(MLContext context, IDataView trainData)
        {
            var trainer = context.Regression.Trainers.Sdca();

            string[] featureColumns = { "Abertura", "Maxima", "Minima", "Volume" };
            IEstimator<ITransformer> pipeline = context.Transforms.CopyColumns("Label", "Fechamento")
                .Append(context.Transforms.Concatenate("Features", featureColumns))
                    .Append(context.Transforms.NormalizeMinMax("Features"))
                        .AppendCacheCheckpoint(context)
                            .Append(trainer);

            ITransformer model = pipeline.Fit(trainData);

            return model;
        }

        private static RegressionMetrics Evaluate(MLContext context, ITransformer model, IDataView testSet)
        {
            IDataView predictions = model.Transform(testSet);
            RegressionMetrics metrics = context.Regression.Evaluate(predictions);
            return metrics;
        }

        private static void VerificarMetricas(RegressionMetrics metrics)
        {
            Console.WriteLine("-------------------- MÉTRICAS --------------------");
            Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError}");
            Console.WriteLine($"Mean Squared Error: {metrics.MeanSquaredError}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");
            Console.WriteLine($"R Squared: {metrics.RSquared}");
            Console.WriteLine("--------------------------------------------------");
        }

        private static void RealizarPrevisao(MLContext context, ITransformer model)
        {
            BTCCSV[] stocks = {
                new BTCCSV
                {
                    Abertura = 47324.700001f,
                    Maxima = 49424.780001f,
                    Minima = 45709.430000f,
                    Fechamento = 47612.450001f,
                    FechamentoAjustado = 47612.450001f,
                    Volume = 36522749952
                },
                new BTCCSV
                {
                    Abertura = 50114.700001f,
                    Maxima = 50205.780001f,
                    Minima = 48725.430000f,
                    Fechamento = 50098.450001f,
                    FechamentoAjustado = 50098.450001f,
                    Volume = 32166727776
                },
                new BTCCSV
                {
                    Abertura = 49354.700001f,
                    Maxima = 50724.780001f,
                    Minima = 48725.430000f,
                    Fechamento = 50098.450001f,
                    FechamentoAjustado = 50098.450001f,
                    Volume = 21939223599
                },
                new BTCCSV
                {
                    Abertura = 47264.700001f,
                    Maxima = 49458.780001f,
                    Minima = 46942.430000f,
                    Fechamento = 49362.450001f,
                    FechamentoAjustado = 49362.450001f,
                    Volume = 25775869261
                }
            };

            PredictionEngine<BTCCSV, BTCPREVISAO> predictor = context.Model.CreatePredictionEngine<BTCCSV, BTCPREVISAO>(model);

            foreach (BTCCSV stock in stocks)
            {
                BTCPREVISAO prediction = predictor.Predict(stock);

                Console.WriteLine("---------------- PREVISÃO ----------------");
                Console.WriteLine($"O preço previsto para o BITCOIN é de R$ {(prediction.Fechamento * 5.8).ToString("N2")}");
                Console.WriteLine($"O preço atual é de R$ {(stock.Fechamento * 5.8).ToString("N2")}");
                Console.WriteLine($"Diferença de R$ {((prediction.Fechamento * 5.8) - (stock.Fechamento * 5.8)).ToString("N2")}");
                Console.WriteLine("------------------------------------------");
            }
        }
    }
}
