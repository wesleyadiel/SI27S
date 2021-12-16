using System;
using Microsoft.ML.Data;

namespace StockForecast.Models
{
    public class BTCPREVISAO
    {
        [ColumnName("Score")]
        public float Fechamento { get; set; }
    }
}