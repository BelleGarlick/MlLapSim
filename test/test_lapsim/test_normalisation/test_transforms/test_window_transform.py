from test_lapsim.test_normalisation.test_transforms.test_transform_base import TestTransformBase


class TestWindowNormalisationTransform(TestTransformBase):

    def test_window_transform_toy(self):
        """Loop through the data and check it's formatted as it should be"""
        normalised_data, partition, transform, inputs, outputs, vehicles = self.load_toy_partition(
            "window", foresight=2, sampling=1)
        self.assertWindowTransform(normalised_data, partition, transform, inputs, outputs, vehicles, 2, 1)

    def test_window_transform_real(self):
        """Test that the windowing works with the correct foresight and sampling
        mapping."""
        normalised_data, partition, transform, inputs, outputs, vehicles = self.load_real_partition(
            "window", foresight=120, sampling=4, cores=4)

        self.assertWindowTransform(normalised_data, partition, transform, inputs, outputs, vehicles, 120, 4)

    def assertWindowTransform(self, normalised_data, partition, transform, x, outputs, vehicles, foresight, sampling):
        total_normals_count = sum([len(normalised_data.widths[x]) for x in range(len(normalised_data))])
        foresight_length = 2 * foresight + 1

        y_pos, y_vel = outputs

        # Check shape is correct
        self.assertListEqual([total_normals_count, 3, foresight_length], list(x.shape))
        self.assertListEqual([total_normals_count, 16], list(vehicles.shape))

        global_normal_index = 0
        for track_index in range(len(normalised_data.angles)):
            for normal_index in range(len(normalised_data.angles[track_index])):
                window = x[global_normal_index]
                win_widths, win_angles, win_offsets, win_vehicle = \
                    window[0], window[1], window[2], vehicles[global_normal_index]

                # Test foresight is correct
                for i in range(foresight_length):
                    target_index = (i + normal_index - foresight) % len(normalised_data.widths[track_index])
                    self.assertFloatEqual(win_widths[i], normalised_data.widths[track_index][target_index])
                    self.assertFloatEqual(win_angles[i], normalised_data.angles[track_index][target_index])
                    self.assertFloatEqual(win_offsets[i], normalised_data.offsets[track_index][target_index])

                # Test vehicle is correct
                for v_index in range(len(win_vehicle)):
                    self.assertFloatEqual(normalised_data.vehicles[track_index][v_index], win_vehicle[v_index])

                global_normal_index += 1

        self.assertSamplingCorrect(normalised_data, y_pos, y_vel, sampling)

        # Testing detransform
        global_normal_index = 0
        for i in range(len(normalised_data)):
            track_length = len(normalised_data.angles[i])
            pred_pos, pred_vel = transform.detransform_and_denormalise(
                track_length=track_length,
                position=outputs[0][global_normal_index:global_normal_index + track_length],
                velocity=outputs[1][global_normal_index:global_normal_index + track_length]
            )
            self.assertFloatListEqual(pred_pos, partition.positions[i])
            self.assertFloatListEqual(pred_vel, partition.velocities[i])
            global_normal_index += track_length
