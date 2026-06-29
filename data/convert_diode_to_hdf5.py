import os
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob

def convert_diode_to_hdf5(diode_root, output_file, max_scenes=10, max_scans=5):
    """
    Конвертирует DIODE в HDF5 формат.
    
    Структура DIODE:
    diode_root/
      scene_00007/
        scan_00082/
          00007_00082_outdoor_000_010.png          ← RGB
          00007_00082_outdoor_000_010_depth.npy    ← depth (большой, ~1900 КБ)
          00007_00082_outdoor_000_010_depth_....npy ← depth (маленький, ~40 КБ)
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    image_files = []
    depth_files = []
    
    # Ищем все сцены (папки scene_*)
    scene_dirs = sorted(glob.glob(os.path.join(diode_root, "scene_*")))[:max_scenes]
    print(f"📊 Найдено сцен: {len(scene_dirs)}")
    
    for scene_dir in scene_dirs:
        # Ищем все scan'ы внутри сцены
        scan_dirs = sorted(glob.glob(os.path.join(scene_dir, "scan_*")))[:max_scans]
        
        for scan_dir in scan_dirs:
            # Находим все PNG файлы (RGB)
            png_files = sorted(glob.glob(os.path.join(scan_dir, "*.png")))
            
            for png_file in png_files:
                base_name = os.path.basename(png_file).replace('.png', '')
                
                # Ищем все depth файлы с таким же именем
                depth_candidates = glob.glob(os.path.join(scan_dir, f"{base_name}_depth*.npy"))
                
                if not depth_candidates:
                    continue
                
                # Берём САМЫЙ БОЛЬШОЙ файл (это dense depth, не sparse)
                depth_candidates.sort(key=os.path.getsize, reverse=True)
                depth_file = depth_candidates[0]
                
                image_files.append(png_file)
                depth_files.append(depth_file)
    
    print(f"📊 Найдено {len(image_files)} пар RGB-Depth")
    
    if len(image_files) == 0:
        print("❌ Файлы не найдены! Проверь путь.")
        return 0
    
    # Конвертируем в HDF5
    with h5py.File(output_file, 'w') as h5f:
        for i, (img_path, depth_path) in enumerate(tqdm(zip(image_files, depth_files), desc="Converting")):
            try:
                rgb_img = Image.open(img_path)
                rgb_array = np.array(rgb_img)
                
                depth_array = np.load(depth_path).astype(np.float32)
                
                group = h5f.create_group(f'sample_{i:06d}')
                group.create_dataset('rgb', data=rgb_array, compression='gzip')
                group.create_dataset('depth', data=depth_array, compression='gzip')
            except Exception as e:
                print(f"️ Ошибка: {img_path} → {e}")
                continue
    
    print(f"✅ Сохранено в {output_file}")
    return len(image_files)


if __name__ == '__main__':
    # ПУТЬ К СЦЕНАМ (у тебя они лежат прямо в diode_raw)
    DIODE_ROOT = "data/diode_raw"
    OUTPUT_FILE = "data/diode_hdf5/diode_train.hdf5"
    
    # Для теста: 3 сцены, по 3 scan'а
    count = convert_diode_to_hdf5(DIODE_ROOT, OUTPUT_FILE, max_scenes=3, max_scans=3)
    print(f"\n📦 Итого: {count} изображений")