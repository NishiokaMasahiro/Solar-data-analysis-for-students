"""
=============================
Co-align IRIS SJI  to SDO/AIA
=============================

In this example we will show how to co-align a rolled IRIS dataset to SDO/AIA.

The IRIS team at LMSAL provides AIA data cubes which are coaligned to the IRIS FOV
for each observation from https://iris.lmsal.com/search/

Therefore this example is more a showcase of functionally.
"""

import matplotlib.pyplot as plt
import numpy as np
import pooch
from sunkit_image.coalignment import coalign

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta

import sunpy.map
from aiapy.calibrate import update_pointing
from aiapy.calibrate.utils import get_pointing_table
from sunpy.net import Fido
from sunpy.net import attrs as a

from irispy.io import read_files

###############################################################################
# IRIS アーカイブからデータをダウンロード。
# 自己完結型に保つために ``pooch`` を使用するが、ブラウザを使用しても動作。

sji_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2025/07/10/20250710_121126_3893010094/iris_l2_20250710_121126_3893010094_SJI_2832_t000_deconvolved.fits.gz",
    known_hash="0875acc65711969a93ce67474b6236bc98ce5a2bc49901ccad0f70ccf1478033",
)

###############################################################################
# slit-jaw imager (SJI) 画像の読み込み。

sji_2832 = read_files(sji_filename)
(time_sji,) = sji_2832.axis_world_coords("time")
sji_map = sji_2832.to_maps(8)

###############################################################################
# VSO から最も近い AIA 170 nm 画像をダウンロード。

search_results = Fido.search(
    a.Time(time_sji[0], Time(time_sji[0]) + TimeDelta(1 * u.minute), near=time_sji[0]),
    a.Instrument.aia,
    a.Wavelength(1700 * u.AA),
)
files = Fido.fetch(search_results, site="NSO")
aia_map = sunpy.map.Map(files[0])
pointing_table = get_pointing_table(
    source="JSOC",
    time_range=(Time(time_sji[0]) - TimeDelta(5 * 60 * u.minute), Time(time_sji[0]) + TimeDelta(1 * u.minute)),
)
aia_map = update_pointing(aia_map, pointing_table=pointing_table)

# AIA FOV を IRIS と同程度、かつより大きくトリミング（完全なカバー範囲を確保）。
# 予想されるシフトと同じ大きさにしないと、エッジ効果が発生。
aia_crop = aia_map.submap(
    bottom_left=SkyCoord(
        sji_map.bottom_left_coord.Tx - 50 * u.arcsec,
        sji_map.bottom_left_coord.Ty - 50 * u.arcsec,
        frame="helioprojective",
        observer=sji_map.bottom_left_coord.observer,
    ),
    top_right=SkyCoord(
        sji_map.top_right_coord.Tx + 50 * u.arcsec,
        sji_map.top_right_coord.Ty + 50 * u.arcsec,
        frame="helioprojective",
        observer=sji_map.top_right_coord.observer,
    ),
)

# ###############################################################################
# IRIS SJI 画像上に AIA 等高線をプロット。
# 以下では ``sunkit-image`` を使用
#
# 画像を位置合わせする前に、まずAIA画像をIRIS画像と同じプレートスケールにリサンプリング。
# これにより、共位置合わせの精度が向上します。
nx = (aia_crop.scale.axis1 * aia_crop.dimensions.x) / sji_map.scale.axis1.to(u.arcsec / u.pix)
ny = (aia_crop.scale.axis2 * aia_crop.dimensions.y) / sji_map.scale.axis2.to(u.arcsec / u.pix)
aia_upsampled = aia_crop.resample(u.Quantity([nx, ny]))

# NaN部分を除去したSJIを用意.
sji_map_corrected_data = sji_map.data.copy()
nan_mask = ~np.isfinite(sji_map_corrected_data)
if np.any(nan_mask):
    sji_map_corrected_data[nan_mask] = 0
sji_map_corrected = sunpy.map.Map(sji_map_corrected_data, sji_map.meta)

coaligned_sji_map = coalign(sji_map_corrected, aia_upsampled, method="match_template")

# ###############################################################################
# 位置合わせ結果のプロット。

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection=sji_map.wcs)
sji_map.plot(axes=ax1)
aia_upsampled.draw_contours(axes=ax1, levels=[500] * u.DN, colors=["red"], linewidths=2)
ax1.set_title("IRIS SJI with AIA contours")

ax2 = fig.add_subplot(122, projection=coaligned_sji_map.wcs, sharex=ax1, sharey=ax1)
coaligned_sji_map.plot(axes=ax2)
aia_upsampled.draw_contours(axes=ax2, levels=[500] * u.DN, colors=["red"], linewidths=2)
ax2.set_title("Co-aligned IRIS SJI with AIA contours")
ax2.coords[1].set_ticks_visible(False)
ax2.coords[1].set_ticklabel_visible(False)

xlims_world = [-570, -490] * u.arcsec
ylims_world = [-210, -140] * u.arcsec
world_coords = SkyCoord(Tx=xlims_world, Ty=ylims_world, frame=coaligned_sji_map.coordinate_frame)
pixel_coords_x, pixel_coords_y = coaligned_sji_map.wcs.world_to_pixel(world_coords)
ax2.set_xlim(pixel_coords_x)
ax2.set_ylim(pixel_coords_y)

fig.tight_layout()

plt.show()
