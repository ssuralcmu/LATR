python shounak_scripts/infer_custom.py   --config config/release_iccv/once.py   --checkpoint pretrained_models/once.pth   --input-dir /home/nishad/Documents/shounak_real_world_lane_boundary/dataset/WorkZone3D/image_1  --output-dir workzone3d_image1_lanes_once

python shounak_scripts/infer_custom.py   --config config/release_iccv/apollo_standard.py   --checkpoint pretrained_models/apollo_standard.pth   --input-dir /home/nishad/Documents/shounak_real_world_lane_boundary/dataset/WorkZone3D/image_1/  --output-dir workzone3d_image1_lanes_apollo

python shounak_scripts/infer_custom.py   --config config/release_iccv/latr_1000_baseline.py   --checkpoint pretrained_models/openlane.pth   --input-dir /home/nishad/Documents/shounak_real_world_lane_boundary/dataset/WorkZone3D/image_1/  --output-dir workzone3d_image1_lanes_openlane

# Build future ego-pose labels from vehicle-state JSONs

python shounak_scripts/generate_future_ego_poses.py \
  --input-dir /home/nishad/Documents/shounak_real_world_lane_boundary/dataset/WorkZone3D/vehicle_state \
  --output-dir /home/nishad/Documents/shounak_real_world_lane_boundary/dataset/WorkZone3D/future_poses \
  --horizon-seconds 10 \
  --timestamp-source vehicle_timestamp

# Plot future trajectories on images using KITTI calib

python shounak_scripts/plot_future_trajectories_on_images.py \
  --future-json-dir /home/nishad/Documents/shounak_real_world_lane_boundary/dataset/WorkZone3D/future_poses \
  --image-dir /home/nishad/Documents/shounak_real_world_lane_boundary/dataset/WorkZone3D/image_1 \
  --calib-dir /home/nishad/Documents/shounak_real_world_lane_boundary/dataset/WorkZone3D/calib \
  --output-dir /home/nishad/Documents/shounak_real_world_lane_boundary/dataset/WorkZone3D/future_traj_on_image

# Fit cubic polylines from future ego-pose JSONs

python shounak_scripts/fit_future_pose_polylines.py \
  --future-json-dir /home/nishad/Documents/shounak_real_world_lane_boundary/dataset/WorkZone3D/future_poses \
  --output-dir /home/nishad/Documents/shounak_real_world_lane_boundary/dataset/WorkZone3D/future_poses_polylines \
  --max-distance-m 20

# Fit cubic polylines from lane marker JSONs
python shounak_scripts/fit_lane_boundary_polylines.py \
--input-dir /home/nishad/Documents/shounak_real_world_lane_boundary/LATR/workzone3d_image1_lanes_once/pred \
--output-dir /home/nishad/Documents/shounak_real_world_lane_boundary/LATR/workzone3d_image1_lanes_once/polylines 

# Fit cubic polylines from work zone object annotations
python shounak_scripts/fit_workzone_object_polylines.py \
  --input-dir /home/nishad/Documents/shounak_real_world_lane_boundary/dataset/WorkZone3D/label_2 \
  --output-dir /home/nishad/Documents/shounak_real_world_lane_boundary/dataset/WorkZone3D/workzone_polylines \
  --class-names Channelizer Cone Barrel