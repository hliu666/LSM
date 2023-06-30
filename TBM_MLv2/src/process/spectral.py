import numpy as np
import joblib
from scipy.interpolate import interp1d

def resample_hyperspectral(data, start=400, end=2500, step=1, new_step=5, discard_ranges=[(1301, 1449), (1751, 1999), (2401, 2500)]):
    # Create initial and new wavelength arrays
    initial_wavelength = np.arange(start, end+step, step)
    new_wavelength = np.arange(start, end+new_step, new_step)

    # Remove wavelengths in discard_ranges from new_wavelength
    for discard_range in discard_ranges:
        new_wavelength = new_wavelength[(new_wavelength < discard_range[0]) | (new_wavelength > discard_range[1])]

    # Initialize new_data depending on dimensions of input data
    if len(data.shape) == 3:
        # Resample each pixel's spectrum
        height, width, _ = data.shape
        new_data = np.zeros((height, width, len(new_wavelength)))
        for i in range(height):
            for j in range(width):
                # Get the spectrum at the current pixel
                spectrum = data[i, j, :]

                # Create an interpolation function for the spectrum
                interp_spectrum = interp1d(initial_wavelength, spectrum, kind='linear', fill_value='extrapolate')

                # Interpolate the spectrum at the new wavelengths
                new_spectrum = interp_spectrum(new_wavelength)

                # Add the new spectrum to the new data array
                new_data[i, j, :] = new_spectrum
    elif len(data.shape) == 2:
        # Resample each spectrum
        num_spectra, _ = data.shape
        new_data = np.zeros((num_spectra, len(new_wavelength)))
        for i in range(num_spectra):
            # Get the spectrum at the current position
            spectrum = data[i, :]

            # Create an interpolation function for the spectrum
            interp_spectrum = interp1d(initial_wavelength, spectrum, kind='linear', fill_value='extrapolate')

            # Interpolate the spectrum at the new wavelengths
            new_spectrum = interp_spectrum(new_wavelength)

            # Add the new spectrum to the new data array
            new_data[i, :] = new_spectrum
    else:
        raise ValueError('Data must be a 2D or 3D numpy array')

    return new_data, new_wavelength

spectral_hourly = joblib.load("../../data/mod_list_spectral.pkl")
spectral_hourly_resample, wavelength_resample = resample_hyperspectral(spectral_hourly)
joblib.dump(spectral_hourly_resample, "../../data/mod_list_spectral_resample.pkl")
joblib.dump(wavelength_resample, "../../data/mod_list_wavelength_resample.pkl")

prospectpro = np.loadtxt("../model/support/dataSpec_PDB.txt")
prospectpro_resample, _ = resample_hyperspectral(prospectpro.transpose())
prospectpro_resample = prospectpro_resample.transpose()
np.savetxt('../model/support/dataSpec_PDB_resample.txt', prospectpro_resample.reshape(-1, prospectpro_resample.shape[-1]), fmt='%.6f', delimiter='\t')
print("test")

