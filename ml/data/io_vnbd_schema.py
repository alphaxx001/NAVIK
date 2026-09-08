import pandas as pd

class IOVNBDSchemaResolver:
    @staticmethod
    def _find_col(cols_orig, keywords, required_name):
        for c_orig in cols_orig:
            c_upper = str(c_orig).strip().upper()
            if all(k.upper() in c_upper for k in keywords):
                return c_orig
        raise ValueError(f"Missing required field '{required_name}'. Available columns: {cols_orig}")

    @staticmethod
    def resolve_imu_columns(df):
        cols = list(df.columns)
        cols_lower = {str(c).strip().lower(): c for c in cols}
        
        # Check direct lowercase matches first
        t_col = cols_lower.get('timestamp') or cols_lower.get('time')
        ax = cols_lower.get('ax') or cols_lower.get('accel_x')
        ay = cols_lower.get('ay') or cols_lower.get('accel_y')
        az = cols_lower.get('az') or cols_lower.get('accel_z')
        gx = cols_lower.get('gx') or cols_lower.get('gyro_x')
        gy = cols_lower.get('gy') or cols_lower.get('gyro_y')
        gz = cols_lower.get('gz') or cols_lower.get('gyro_z')

        if not (t_col and ax and ay and az and gx and gy and gz):
            t_col = IOVNBDSchemaResolver._find_col(cols, ['TIME'], 'timestamp')
            ax = IOVNBDSchemaResolver._find_col(cols, ['ACCEL', 'X'], 'accelerometer X')
            ay = IOVNBDSchemaResolver._find_col(cols, ['ACCEL', 'Y'], 'accelerometer Y')
            az = IOVNBDSchemaResolver._find_col(cols, ['ACCEL', 'Z'], 'accelerometer Z')
            gx = IOVNBDSchemaResolver._find_col(cols, ['GYRO', 'X'], 'gyroscope X')
            gy = IOVNBDSchemaResolver._find_col(cols, ['GYRO', 'Y'], 'gyroscope Y')
            gz = IOVNBDSchemaResolver._find_col(cols, ['GYRO', 'Z'], 'gyroscope Z')

        return {
            'time': t_col,
            'accel_x': ax,
            'accel_y': ay,
            'accel_z': az,
            'gyro_x': gx,
            'gyro_y': gy,
            'gyro_z': gz
        }
        
    @staticmethod
    def resolve_v_columns(df):
        cols = list(df.columns)
        cols_lower = {str(c).strip().lower(): c for c in cols}

        t_col = cols_lower.get('timestamp') or cols_lower.get('time')
        lat = cols_lower.get('gnss_lat') or cols_lower.get('lat') or cols_lower.get('latitude')
        lon = cols_lower.get('gnss_lon') or cols_lower.get('lon') or cols_lower.get('longitude')
        vel = cols_lower.get('speed') or cols_lower.get('velocity') or cols_lower.get('speed_mps')
        head = cols_lower.get('gnss_head') or cols_lower.get('heading')
        yaw_rate = cols_lower.get('yaw_rate') or cols_lower.get('yaw rate')

        if not (t_col and lat and lon):
            t_col = IOVNBDSchemaResolver._find_col(cols, ['TIME'], 'timestamp')
            lat = IOVNBDSchemaResolver._find_col(cols, ['LAT'], 'GPS latitude')
            lon = IOVNBDSchemaResolver._find_col(cols, ['LON'], 'GPS longitude')
        
        if vel is None:
            try:
                vel = IOVNBDSchemaResolver._find_col(cols, ['VELOCITY'], 'vehicle velocity')
            except ValueError:
                try:
                    vel = IOVNBDSchemaResolver._find_col(cols, ['SPEED'], 'vehicle speed')
                except ValueError:
                    vel = None

        if head is None:
            try:
                head = IOVNBDSchemaResolver._find_col(cols, ['HEADING'], 'vehicle heading')
            except ValueError:
                head = None
            
        if yaw_rate is None:
            try:
                yaw_rate = IOVNBDSchemaResolver._find_col(cols, ['YAW RATE'], 'vehicle yaw rate')
            except ValueError:
                yaw_rate = None
            
        return {
            'time': t_col,
            'lat': lat,
            'lon': lon,
            'velocity': vel,
            'heading': head,
            'yaw_rate': yaw_rate
        }
