cv::Vec2f vc_align_and_extract_umbilicus(const GridStore& grid_store, std::optional<std::uint32_t> seed,
                                      float uncrush) {
    auto segments = grid_store.get_all();
    if (segments.empty()) {
        return cv::Vec2f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
    }

    // 1. Define RANSAC parameters
    const int num_ransac_iterations = 1000;
    const int num_samples_per_iteration = 10000;

    // 2. Determine grid extents for sampling umbilicus candidates
    cv::Size grid_size = grid_store.size();
    float min_x = 0;
    float max_x = grid_size.width;
    float min_y = 0;
    float max_y = grid_size.height;

    // 3. Create a global list of all line segments
    std::vector<std::pair<cv::Point, cv::Point>> all_line_segments;
    for (const auto& path : segments) {
        if (path->size() < 2) continue;
        for (size_t i = 0; i < path->size() - 1; ++i) {
            all_line_segments.push_back(std::make_pair((*path)[i], (*path)[i+1]));
        }
    }

    if (all_line_segments.empty()) {
        return cv::Vec2f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
    }

    // 3b. Optional un-crush. The score below looks for the point the sheet normals point at,
    // which is the centre of round windings but not of crushed ones: the normals of an ellipse
    // do not point at its centre. With uncrush = alpha > 0 the section is first stretched back
    // towards round. The segment midpoints are whitened along their principal axes with
    // strength alpha, x' = W (x - mu) with W = diag((l_i / l_mean)^(-alpha / 2)) V^T, the
    // normals are carried by the inverse transpose of W, the search runs in that frame over the
    // bounds of the whitened midpoints, and its result is mapped back with W^-1. alpha = 1 makes
    // the covariance of the midpoints round; alpha = 0 is the estimator unchanged, draw for draw.
    double w00 = 1.0, w01 = 0.0, w10 = 0.0, w11 = 1.0;   // W, row major
    double mu_x = 0.0, mu_y = 0.0;
    bool whiten = false;
    if (uncrush > 0.0f) {
        const double n = (double)all_line_segments.size();
        for (const auto& s : all_line_segments) {
            mu_x += 0.5 * (s.first.x + s.second.x);
            mu_y += 0.5 * (s.first.y + s.second.y);
        }
        mu_x /= n;
        mu_y /= n;
        double cxx = 0.0, cxy = 0.0, cyy = 0.0;   // MIDPOINT version: covariance of the segment midpoints (reconstructed from extracted_n.inc; bit-identical to the binary used, see README)
        for (const auto& s : all_line_segments) {
            const double dx = 0.5 * (s.first.x + s.second.x) - mu_x;
            const double dy = 0.5 * (s.first.y + s.second.y) - mu_y;
            cxx += dx * dx;
            cxy += dx * dy;
            cyy += dy * dy;
        }
        // Eigen decomposition of the symmetric 2 x 2 covariance, in closed form. The common
        // factor 1/n is left out: only the ratios of the eigenvalues are used.
        const double half_trace = 0.5 * (cxx + cyy);
        const double r = std::hypot(0.5 * (cxx - cyy), cxy);
        const double l1 = half_trace + r, l2 = half_trace - r;
        if (l2 > 0.0) {
            const double t = 0.5 * std::atan2(2.0 * cxy, cxx - cyy);   // angle of the major axis
            const double f1 = std::pow(l1 / half_trace, -0.5 * uncrush);
            const double f2 = std::pow(l2 / half_trace, -0.5 * uncrush);
            w00 = f1 * std::cos(t);  w01 = f1 * std::sin(t);
            w10 = -f2 * std::sin(t); w11 = f2 * std::cos(t);
            whiten = true;
            min_x = min_y = std::numeric_limits<float>::max();
            max_x = max_y = std::numeric_limits<float>::lowest();
            for (const auto& s : all_line_segments) {
                const double dx = 0.5 * (s.first.x + s.second.x) - mu_x;
                const double dy = 0.5 * (s.first.y + s.second.y) - mu_y;
                const float wx = (float)(w00 * dx + w01 * dy), wy = (float)(w10 * dx + w11 * dy);
                min_x = std::min(min_x, wx); max_x = std::max(max_x, wx);
                min_y = std::min(min_y, wy); max_y = std::max(max_y, wy);
            }
        }
    }
    const double det = w00 * w11 - w01 * w10;

    // 4. Pre-sample points and calculate normals for consistent scoring
    std::vector<cv::Point2f> sample_points;
    std::vector<cv::Vec2f> sample_normals;
    sample_points.reserve(num_samples_per_iteration);
    sample_normals.reserve(num_samples_per_iteration);

    // The generator. With no seed it is the one static generator seeded once from the hardware,
    // as published, so two calls in one process draw different samples. With a seed it is a
    // fresh generator seeded with that value, so that a run can be repeated exactly.
    static std::mt19937 shared_gen(std::random_device{}());
    std::mt19937 seeded_gen;
    if (seed) seeded_gen.seed(*seed);
    std::mt19937& gen = seed ? seeded_gen : shared_gen;
    std::uniform_int_distribution<> segment_dist(0, all_line_segments.size() - 1);

    for (int j = 0; j < num_samples_per_iteration; ++j) {
        const auto& line_segment = all_line_segments[segment_dist(gen)];
        const auto& p1 = line_segment.first;
        const auto& p2 = line_segment.second;

        sample_points.emplace_back((float)(p1.x + p2.x) / 2.0f, (float)(p1.y + p2.y) / 2.0f);

        cv::Vec2f tangent((float)(p2.x - p1.x), (float)(p2.y - p1.y));
        cv::normalize(tangent, tangent);
        sample_normals.emplace_back(-tangent[1], tangent[0]); // Perpendicular

        if (whiten) {
            const double dx = sample_points.back().x - mu_x, dy = sample_points.back().y - mu_y;
            sample_points.back() = cv::Point2f((float)(w00 * dx + w01 * dy), (float)(w10 * dx + w11 * dy));
            const cv::Vec2f nrm = sample_normals.back();   // a normal goes through W^-T
            cv::Vec2f wn((float)((w11 * nrm[0] - w10 * nrm[1]) / det),
                         (float)((-w01 * nrm[0] + w00 * nrm[1]) / det));
            cv::normalize(wn, wn);
            sample_normals.back() = wn;
        }
    }

    // 4. RANSAC loop
    cv::Vec2f best_umbilicus(0, 0);
    double best_score = -1.0;

    std::uniform_real_distribution<float> x_dist(min_x, max_x);
    std::uniform_real_distribution<float> y_dist(min_y, max_y);

    for (int i = 0; i < num_ransac_iterations; ++i) {
        // a. Sample a candidate umbilicus point
        cv::Vec2f candidate_umbilicus(x_dist(gen), y_dist(gen));

        // b. Score the candidate
        double current_score = 0.0;
        for (size_t j = 0; j < sample_points.size(); ++j) {
            const auto& point = sample_points[j];
            const auto& normal = sample_normals[j];

            cv::Vec2f umbilicus_to_segment = cv::Vec2f(point) - candidate_umbilicus;
            cv::normalize(umbilicus_to_segment, umbilicus_to_segment);

            double cos_angle = umbilicus_to_segment.dot(normal);
            current_score += cos_angle * cos_angle;
        }

        // c. Update best estimate
        if (current_score > best_score) {
            best_score = current_score;
            best_umbilicus = candidate_umbilicus;
        }
    }

    // 5. Refine the best estimate using a hill-climbing direct search.
    auto score_candidate = [&](const cv::Vec2f& candidate) {
        double score = 0.0;
        for (size_t j = 0; j < sample_points.size(); ++j) {
            const auto& point = sample_points[j];
            const auto& normal = sample_normals[j];

            cv::Vec2f umbilicus_to_segment = cv::Vec2f(point) - candidate;
            float dist = cv::norm(umbilicus_to_segment);
            if (dist < 1e-6) continue;

            umbilicus_to_segment /= dist; // Manual normalization

            double cos_angle = umbilicus_to_segment.dot(normal);
            float weight = 1.0f / std::max(100.0f, dist);
            score += (cos_angle * cos_angle) * weight;
        }
        // Weighted sum, not weighted mean. The sum weights the alignment of each normal by the
        // proximity of its segment, so it favours candidates near many segments as well as
        // candidates the normals point at; it vanishes far from the section and so has a maximum
        // at finite distance, which the mean does not on partial predictions. That maximum is a
        // property of the score and not a guarantee about the search below it: nothing bounds the
        // number of steps the climb takes or keeps it in the grid, and that it settled inside the
        // grid on 576 of 576 measured slices is observed and not enforced. On synthetic
        // sections with a known centre the mean is exact and the sum is biased towards the denser
        // side by 0.4 to 2.4 mm (evidence/synthetic-density.csv of the validation paper). On
        // fifteen real scrolls with a manual reference the sum's median error is 0.4 to 7.1 mm
        // against 0.4 to 133 mm for the mean, with no failure above 10 mm against seven scrolls,
        // and the failure persists with the walk clipped to the grid: it is the objective and
        // not the search.
        return score;
    };

    double refined_best_score = score_candidate(best_umbilicus);
    float step_size = 1024.0f;

    while (step_size >= 1.0f) {
        bool moved = false;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                if (dx == 0 && dy == 0) continue;

                cv::Vec2f candidate = best_umbilicus + cv::Vec2f(dx * step_size, dy * step_size);
                double candidate_score = score_candidate(candidate);

                if (candidate_score > refined_best_score) {
                    refined_best_score = candidate_score;
                    best_umbilicus = candidate;
                    moved = true;
                }
            }
        }

        if (!moved) {
            step_size /= 2.0f;
        }
    }

    if (whiten) {   // back from the un-crushed frame, x = W^-1 x' + mu
        const double ux = best_umbilicus[0], uy = best_umbilicus[1];
        best_umbilicus = cv::Vec2f((float)((w11 * ux - w01 * uy) / det + mu_x),
                                   (float)((-w10 * ux + w00 * uy) / det + mu_y));
    }

    std::cout << "Refined umbilicus estimate: " << best_umbilicus << " with score " << refined_best_score << std::endl;

    return best_umbilicus;
}

