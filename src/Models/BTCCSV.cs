using System;
using Microsoft.ML.Data;

namespace StockForecast.Models
{
    public class BTCCSV
    {
        [LoadColumn(0)]
        public DateTime Data { get; set; }

        [LoadColumn(1)]
        public float Abertura { get; set; }

        [LoadColumn(2)]
        public float Maxima { get; set; }

        [LoadColumn(3)]
        public float Minima { get; set; }

        [LoadColumn(4)]
        public float Fechamento { get; set; }

        [LoadColumn(5)]
        public float FechamentoAjustado { get; set; }

        [LoadColumn(6)]
        public float Volume { get; set; }
    }
}