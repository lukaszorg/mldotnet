using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using otomotoModelBuilder.Annotations;

namespace OtomotoModelBuilder
{
    public class ColumnModel:BaseViewModel
    {
        private ColumnTransform _columnTransform;
        public string Name { get; set; }
        public IEnumerable<ColumnTransform> Transforms { get; internal set; }

        public ColumnTransform SelectedTransform
        {
            get => _columnTransform;
            set => SetProperty(ref _columnTransform, value);
        }
    }


    public class BaseViewModel: INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }


        [NotifyPropertyChangedInvocator]
        protected virtual bool SetProperty<T>(ref T storage, T value, [CallerMemberName] string propertyName = null)
        {
            if (EqualityComparer<T>.Default.Equals(storage, value))
            {
                return false;
            }

            storage = value;
            OnPropertyChanged(propertyName);
            return true;
        }
    }

    public enum ColumnTransform
    {
        None,
        OneHotEncoding,
        OneHotHashEncoding,
        NormalizeMeanVariance,
        NormalizeMinMax,

    }
}