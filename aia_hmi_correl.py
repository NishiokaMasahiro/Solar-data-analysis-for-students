import numpy as np
import matplotlib.pyplot as plt
from sunpy.map import Map
from reproject import reproject_interp
import astropy.units as u
from astropy.coordinates import SkyCoord

def plot_aia_hmi_with_magnetogram(
    hmi_intensity_fits,
    hmi_magnetgram_fits,
    aia_fits,
    x_range=(-295, -225),
    y_range=(35, 70)
):
    """
    AIA, HMI連続光, HMI磁場の3つを同じWCS・視野で並べて可視化。
    磁場はグレースケール・[-300, 300]Gaussで表示。
    """
    hmi_map    = Map(hmi_intensity_fits)
    magnet_map = Map(hmi_magnetgram_fits)
    aia_map    = Map(aia_fits)

    # --- Intensity map reproject ---
    hmi_data_reproj, _ = reproject_interp(hmi_map, aia_map.wcs, shape_out=aia_map.data.shape)
    hmi_map_reproj     = Map(hmi_data_reproj, aia_map.meta)

    # --- Magnetogram map reproject ---
    magnet_data_reproj, _ = reproject_interp(magnet_map, aia_map.wcs, shape_out=aia_map.data.shape)
    magnet_map_reproj     = Map(magnet_data_reproj, aia_map.meta)

    # サブマップ範囲指定
    bottom_left = SkyCoord(x_range[0]*u.arcsec, y_range[0]*u.arcsec, frame=hmi_map_reproj.coordinate_frame)
    top_right   = SkyCoord(x_range[1]*u.arcsec, y_range[1]*u.arcsec, frame=hmi_map_reproj.coordinate_frame)

    hmi_sub    = hmi_map_reproj.submap(bottom_left, top_right=top_right)
    magnet_sub = magnet_map_reproj.submap(bottom_left, top_right=top_right)
    aia_sub    = aia_map.submap(bottom_left, top_right=top_right)
    
    # --- 可視化1 ---
    fig = plt.figure(figsize=(15, 4))

    ax1 = fig.add_subplot(1, 3, 1, projection=hmi_sub)
    hmi_data_squared = np.square(np.clip(hmi_sub.data, a_min=1, a_max=None))
    hmi_sub.plot(axes=ax1, data=hmi_data_squared, cmap='afmhot', vmin=30000, vmax=120000)
    ax1.set_title("HMI Continuum reprojected")

    ax2 = fig.add_subplot(1, 3, 2, projection=magnet_sub)
    magnet_sub.plot(axes=ax2, cmap='gray', vmin=-300, vmax=300)
    ax2.set_title("HMI Magnetogram reprojected")

    ax3 = fig.add_subplot(1, 3, 3, projection=aia_sub)
    aia_sub.plot(axes=ax3, data=aia_sub.data, vmin=200, vmax=3000)
    ax3.set_title("AIA 1600")
    
    plt.tight_layout()
    plt.savefig("aia_hmi_20130924_115700.png", dpi=150)
    plt.show()
    
    # --- 可視化2 ---
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection=aia_sub)
    aia_sub.plot(axes=ax, data=aia_sub.data, vmin=200, vmax=3000)
    aia_sub.draw_grid(axes=ax)
    ax.set_title("AIA 1600 with HMI magnetic field", y=1.5)

    # HMI磁場のcontourレベルはAstropy Quantity（Gauss）で渡す
    levels = [30, 50, 150, 300] * u.Gauss
    levels = np.concatenate((-1 * levels[::-1], levels))
    bounds = ax.axis()

    # bunitを'G'にしたMapを作成（1DN=1G前提）
    magnet_sub_g = Map(magnet_sub.data, magnet_sub.meta.copy())
    magnet_sub_g.meta['bunit'] = 'G'

    cset = magnet_sub_g.draw_contours(levels, axes=ax, cmap='seismic', alpha=0.5)
    ax.axis(bounds)
    plt.colorbar(cset, label="Magnetic Field Strength [Gauss]", ticks=list(levels.value) + [0], shrink=0.8, pad=0.17)
    plt.savefig("aia_hmi_20130924_115700_magnetogram_contours.png", dpi=150)
    plt.show()
    
# --- サンプル実行例 ---
if __name__ == "__main__":
    hmi_intensity_fits  = "2013_09_24_fits/hmi_data/hmi.Ic_45s.20130924_115700_TAI.2.continuum.fits"
    hmi_magnetgram_fits = "2013_09_24_fits/hmi_data/hmi.M_45s.20130924_115700_TAI.2.magnetogram.fits"
    aia_fits            = "2013_09_24_fits/aia_data/aia.lev1_uv_24s.2013-09-24T115706Z.1600.image_lev1.fits"

    plot_aia_hmi_with_magnetogram(hmi_intensity_fits, hmi_magnetgram_fits, aia_fits)

