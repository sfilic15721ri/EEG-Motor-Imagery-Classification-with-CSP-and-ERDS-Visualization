import mne
from mne.decoding import CSP
from mne.datasets import eegbci
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

# === 1. UČITAVANJE I PREPROCESIRANJE PODATAKA ===

subject = 1
runs = [3, 7]  # 3 = motor imagination (hands), 7 = rest

# Preuzimanje podataka i učitavanje
file_paths = eegbci.load_data(subject, runs, update_path=True)
raws = [mne.io.read_raw_edf(f, preload=True) for f in file_paths]
raw = mne.concatenate_raws(raws)

# Preimenovanje kanala zbog monture
raw.rename_channels(lambda x: x.strip('.').upper())

# Postavljanje standardne monture
montage = mne.channels.make_standard_montage("standard_1005")
raw.set_montage(montage, on_missing='ignore')

# Filtracija signala
raw.filter(7., 30., fir_design='firwin')

# === 2. IZVLAČENJE DOGAĐAJA I EPOHA ===

events, event_id = mne.events_from_annotations(raw)

# Izdvajanje epoha između 1 i 4 sekunde (tokom zadatka)
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=1, tmax=4,
                    baseline=None, preload=True)

X = epochs.get_data()
y = epochs.events[:, 2]  # Oznake klasa

# === 3. CSP + LDA KLASIFIKACIJA ===

csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
clf = make_pipeline(csp, LinearDiscriminantAnalysis())

scores = cross_val_score(clf, X, y, cv=5, n_jobs=1)
print(f"Prosečna tačnost klasifikacije: {np.mean(scores) * 100:.2f}%")

# === 4. ERDS VIZUALIZACIJA ===

# Koristi iste epohe (0–4 s) za ERDS mapu
epochs_erds = mne.Epochs(raw, events, event_id=event_id, tmin=0, tmax=4,
                         baseline=None, preload=True)

freqs = np.arange(8., 30., 2)  # Alfa i beta opseg
n_cycles = freqs / 2.  # Različit broj ciklusa po frekvenciji

# Računanje vremensko-frekvencijske reprezentacije
power = mne.time_frequency.tfr_morlet(epochs_erds, freqs=freqs, n_cycles=n_cycles,
                                      use_fft=True, return_itc=False,
                                      decim=3, n_jobs=1)

# Prikaz ERDS mape za jedan kanal (npr. 'C3' – relevantan za motorni zadatak)
power.plot(picks='C3', baseline=(None, 0), mode='percent', title='ERDS mapa – kanal C3')
plt.show()


