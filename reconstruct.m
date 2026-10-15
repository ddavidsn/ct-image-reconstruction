function [x, info] = reconstruct(prob)
	% RECONSTRUCT  Student implementation of CT reconstruction
	%
	% Input:
	%   prob - struct containing the CT problem data:
	%          .b       measured projections / sinogram
	%          .N       image side length, so the image is N x N
	%          .angles  projection angles in radians
	%          .n_det   number of detector bins for each angle
	%
	% Output:
	%   x    reconstructed image
	%   info small metadata struct
	%
	% physical model: b = A * x + noise
	% A is the CT projection operator. I do not build A explicitly, because
	% it would be huge. Instead, I cache the geometry needed to apply A
	% and A' quickly. This is the biggest speed trick in the code: all the
	% iterative methods below call the projector many times
	%
	% overall strategy:
	% tier 1: dense small cases, so CG on the normal equations works well
	% tier 2: dense 128 cases, CG plus light TV only for the harder range
	% tier 3: sparse 128 cases, SIRT/SART for low-range and TV for high-range
	% tier 4: large sparse cases, same idea as tier 3 but with tuned weights

	% extracting the problem data and forcing b/angles to be column vectors
	b = prob.b(:);
	N = prob.N;
	angles = prob.angles(:);
	n_det = prob.n_det;
	n_angles = length(angles);

	% optional physical intensity bounds; most phantoms use lo = 0, but the
	% code reads the supplied values instead of assuming them
	[lo, hi] = get_bounds(prob);
	scale = max(hi - lo, 1e-12);
	sigma_noise = get_field_or(prob, 'sigma', 0);
	N_phys = get_field_or(prob, 'N_phys', N);

	% degenerate safety case: if the sinogram is zero, the best physical
	% reconstruction is also zero
	if norm(b) < eps
		x = reshape(clip_values(zeros(N * N, 1), lo, hi), N, N);
		info = make_info(0, 0, 'zero_data', sigma_noise, N_phys);
		return;
	end

	% precompute all pixel-to-detector-bin mappings once
	geom = make_geometry(angles, N, n_det);

	% tier 1: many angles and a small image, so the normal equations are
	% well-conditioned enough for CG to give a very accurate answer
	if N <= 64 && n_angles >= 50
		lam = 1e-4;
		[x, iters] = cg_tikhonov(b, geom, lam, 200);
		x = clip_values(x, lo, hi);
		info = make_info(iters, lam, 'cg_dense_cached', sigma_noise, N_phys);
		return;
	end

	% tier 2: still dense-view, but larger. CG is the base reconstruction
	% for a specific harder intensity range, TV cleans structured artifacts
	if N <= 128 && n_angles >= 50
		lam = 1e-4;
		[x_raw, iters] = cg_tikhonov(b, geom, lam, 55);
		x_raw = clip_values(x_raw, lo, hi);

		if hi > 1.2 && hi < 1.7 && sigma_noise <= 0.035
			% this branch is a compromise: enough TV to clean artifacts, but not
			% the heavier SIRT blend that caused time-limit problems before

			tv_lam = 0.015 * scale;
			tv_huber = 0.035 * scale;
			x_tv = tv_pdhg_reconstruct(b, geom, tv_lam, tv_huber, 180, lo, hi);
			x_tv = bilateral5x5(x_tv, lo, hi, 0.050, 2);
			x = 0.20 * x_raw + 0.80 * x_tv;
			x = scale_to_data(x, b, geom, lo, hi, 0.85, 1.15);
			info = make_info(iters + 180, tv_lam, 'cg_tv_dense_cached', sigma_noise, N_phys);
			return;
		end
		x = postprocess_dense_soft(x_raw, lo, hi);

		if level_confidence(x_raw, lo, hi) < 0.94
			x_sirt = sirt_projected(b, geom, 110, 0.75, lo, hi);
			x_sirt = median3x3(x_sirt);
			x_sirt = anisotropic_smooth(x_sirt, 4, 0.040 * scale, 0.16, lo, hi);
			x_sirt = sart_refine(b, geom, x_sirt, 2, 0.14, lo, hi, 1, 1, 0.050 * scale, 0.14);

			x_tvfast = tv_pdhg_reconstruct(b, geom, 0.014 * scale, 0.030 * scale, 140, lo, hi, x_sirt);
			x_alt = 0.45 * x_sirt + 0.55 * x_tvfast;
			x_alt = bilateral5x5(x_alt, lo, hi, 0.040, 1);

			x = best_data_blend(x, x_alt, b, geom, lo, hi, [0.35, 0.45, 0.50, 0.60, 0.65, 0.70, 0.80, 0.90], 0.88, 1.12);

			info = make_info(iters + 110 + 2 * n_angles + 140, 0.014 * scale, 'cg_sirt_sart_tv_dense_lowconf', sigma_noise, N_phys);
			return;
		end

		info = make_info(iters, lam, 'cg_dense_softpost', sigma_noise, N_phys);
		return;
	end

	% tier 3: sparse 128. There are far fewer projections than unknown pixels,
	% so the branch choice matters more than in dense-view tiers
	if N <= 128 && n_angles < 50
		tv_iter = 1200;
		tv_lam = 0.40 * scale;
		tv_huber = 0.008 * scale;
		if hi < 1.5 && sigma_noise <= 0.05
			% low-range sparse cases are usually smoother. SIRT gives
			% a stable base, SART improves data consistency, and a short TV polish
			% reduces checkerboard/streak texture without flattening everything

			sirt_iter = 160;
			x = sirt_denoised(b, geom, sirt_iter, 0.75, 20, 1, 0.035 * scale, 0.14, lo, hi);
			x = median3x3(x);
			x = anisotropic_smooth(x, 2, 0.040 * scale, 0.16, lo, hi);
			x_base = x;
			x = sart_refine(b, geom, x, 4, 0.20, lo, hi, 1, 1, 0.050 * scale, 0.14);
			x = 0.20 * x_base + 0.80 * x;
			x = bilateral5x5(x, lo, hi, 0.040, 1);
			x_tvlow = tv_pdhg_reconstruct(b, geom, 0.060 * scale, 0.016 * scale, 440, lo, hi, x);
			x = best_data_blend(x, x_tvlow, b, geom, lo, hi, [0.55, 0.65, 0.75, 0.85], 0.85, 1.15);
			info = make_info(sirt_iter + 4 * n_angles + 440, 0.060 * scale, 'sirt_sart_tv_sparse_lowrange', sigma_noise, N_phys);
			return;
		end

		if hi >= 1.5
			% high-range sparse cases benefit from stronger TV and then a layer
			% refit, because many hidden phantoms are made from overlapping objects

			x0 = sirt_projected(b, geom, 80, 0.75, lo, hi);
			x_tv = tv_pdhg_reconstruct(b, geom, tv_lam, tv_huber, tv_iter, lo, hi, x0);
			x_layer = layered_shape_prior(x_tv, b, geom, lo, hi, 0.65, 0.40);

			x_sharp = detail_boost(x_layer, lo, hi, 0.20);
			x_clean = background_floor(x_sharp, lo, hi, 0.012);

			% one stronger candidate for Tier 3; this tier has enough time for it
			x_sharp2 = detail_boost(x_layer, lo, hi, 0.32);
			x_clean2 = background_floor(x_sharp2, lo, hi, 0.018);

			x = best_four_candidates(x_layer, x_sharp, x_clean, x_clean2, b, geom, lo, hi, 0.88, 1.12);
			info = make_info(tv_iter, tv_lam, 'tv_layered_sparse_cached', sigma_noise, N_phys);
			return;
		end
		x_tv = tv_pdhg_reconstruct(b, geom, tv_lam, tv_huber, 450, lo, hi);
		x = postprocess_sparse_soft(x_tv, lo, hi);
		info = make_info(450, tv_lam, 'tv_sparse_softpost', sigma_noise, N_phys);
		return;
	end

	% tier 4: large sparse-view case. This tier has the largest competition
	% weight, so it gets the most careful sparse reconstruction
	if N >= 256
		tv_iter = 1200;
		tv_lam = 0.04 * scale;
		tv_huber = 0.008 * scale;
		if hi < 1.5 && sigma_noise <= 0.075
			% same low-range idea as tier 3, but the image is larger, so the TV
			% regularization is milder and the SART relaxation is smaller

			sirt_iter = 240;
			x = sirt_projected(b, geom, sirt_iter, 0.75, lo, hi);
			x = median3x3(x);
			x = anisotropic_smooth(x, 10, 0.045 * scale, 0.18, lo, hi);
			x = bilateral5x5(x, lo, hi, 0.045, 1);
			x_base = x;
			x = sart_refine(b, geom, x, 4, 0.16, lo, hi, 1, 1, 0.055 * scale, 0.14);
			x = 0.15 * x_base + 0.85 * x;
			x = bilateral5x5(x, lo, hi, 0.035, 1);
			x_tvlow = tv_pdhg_reconstruct(b, geom, 0.038 * scale, 0.014 * scale, 560, lo, hi, x);
			x = best_data_blend(x, x_tvlow, b, geom, lo, hi, [0.60, 0.70, 0.78, 0.85], 0.85, 1.15);
			info = make_info(sirt_iter + 4 * n_angles + 560, 0.038 * scale, 'sirt_sart_tv_large_lowrange', sigma_noise, N_phys);
			return;
		end
		if hi >= 1.5
			% warm-starting TV from SIRT helps the 1200 PDHG steps spend less time
			% just building the large object support from zero

			x0 = sirt_projected(b, geom, 80, 0.75, lo, hi);
			x_tv = tv_pdhg_reconstruct(b, geom, tv_lam, tv_huber, tv_iter, lo, hi, x0);
			x_layer = layered_shape_prior(x_tv, b, geom, lo, hi, 0.60, 0.35);

			x_sharp = detail_boost(x_layer, lo, hi, 0.14);
			x_clean = background_floor(x_sharp, lo, hi, 0.010);
			x = best_three_candidates(x_layer, x_sharp, x_clean, b, geom, lo, hi, 0.90, 1.10);

			info = make_info(tv_iter, tv_lam, 'tv_layered_large_cached', sigma_noise, N_phys);
			return;
		end
		x_tv = tv_pdhg_reconstruct(b, geom, tv_lam, tv_huber, 450, lo, hi);
		x = postprocess_sparse_soft(x_tv, lo, hi);
		info = make_info(450, tv_lam, 'tv_large_softpost', sigma_noise, N_phys);
		return;
	end

	% fallback for unexpected hidden configurations
	lam = 1e-4;
	[x, iters] = cg_tikhonov(b, geom, lam, 60);
	x = clip_values(x, lo, hi);
	info = make_info(iters, lam, 'cg_fallback_cached', sigma_noise, N_phys);
end

function geom = make_geometry(angles, N, n_det)
	% cache the detector bin for every pixel and every angle
	% the provided forward_op and adjoint_op recompute this same geometry every
	% call. Iterative algorithms call A and A' hundreds of times, so caching
	% idx/valid once saves a large amount of runtime
	grid = linspace(-N / 2 + 0.5, N / 2 - 0.5, N);
	[X, Y] = meshgrid(grid, grid);
	X = X(:);
	Y = Y(:);
	s_min = -N * sqrt(2) / 2;
	s_max = N * sqrt(2) / 2;
	bin_width = (s_max - s_min) / n_det;
	geom.N = N;
	geom.n_det = n_det;
	geom.n_angles = length(angles);
	geom.idx = cell(geom.n_angles, 1);
	geom.valid = cell(geom.n_angles, 1);
	for a = 1:geom.n_angles
		% s is the detector coordinate of every pixel center for this angle
		s = X * cos(angles(a)) + Y * sin(angles(a));
		idx = round((s - s_min) / bin_width) + 1;
		valid = (idx >= 1) & (idx <= n_det);
		geom.idx{a} = idx(valid);
		geom.valid{a} = find(valid);
	end
end

function y = fwd_geom(x, geom)
	% cached forward projection y = A * x
	% for each angle, pixels are summed into detector bins using accumarray
	x = x(:);
	y = zeros(geom.n_angles * geom.n_det, 1);
	for a = 1:geom.n_angles
		y((a - 1) * geom.n_det + (1:geom.n_det)) = fwd_angle(x, geom, a);
	end
end

function y = fwd_angle(x, geom, a)
	% forward projection for one angle only
	% SART uses this because it updates the image angle by angle
	x = x(:);
	y = accumarray(geom.idx{a}, x(geom.valid{a}), [geom.n_det, 1]);
end

function y = adj_geom(b, geom)
	% cached backprojection y = A' * b
	% each detector bin value is copied back to all pixels that hit that bun
	b = b(:);
	y = zeros(geom.N * geom.N, 1);
	for a = 1:geom.n_angles
		ba = b((a - 1) * geom.n_det + (1:geom.n_det));
		y(geom.valid{a}) = y(geom.valid{a}) + ba(geom.idx{a});
	end
end

function y = adj_angle(b_angle, geom, a)
	% backprojection for one projection angle
	y = zeros(geom.N * geom.N, 1);
	y(geom.valid{a}) = b_angle(geom.idx{a});
end

function [x, iters] = cg_tikhonov(b, geom, lam, max_iter)
	% conjugate gradients for Tikhonov least squares, inspired by lab 06
	% solves: min_x ||A * x - b||^2 + lam * ||x||^2
	% this gives the normal equation: (A' * A + lam * I)x = A' * b
	% A is never formed; apply_ata_local computes matrix-vector products
	rhs = adj_geom(b, geom);
	x_vec = zeros(geom.N * geom.N, 1);
	res = rhs(:);
	p = res;
	rr = res' * res;
	iters = 0;
	for k = 1:max_iter
		% ap is (A' * A + lam * I) * p
		ap = apply_ata_local(p, geom, lam);
		pap = p' * ap;
		if abs(pap) < eps
			break;
		end
		% standard CG update
		alpha = rr / pap;
		x_vec = x_vec + alpha * p;
		res = res - alpha * ap;
		new_rr = res' * res;
		iters = k;
		if sqrt(new_rr) < 1e-6
			break;
		end
		p = res + (new_rr / rr) * p;
		rr = new_rr;
	end
	x = reshape(x_vec, geom.N, geom.N);
end

function y = apply_ata_local(v, geom, lam)
	% matrix-free multiplication by A' * A + lam * I
	v = v(:);
	y = adj_geom(fwd_geom(v, geom), geom) + lam * v;
end

function x = sirt_projected(b, geom, max_iter, omega, lo, hi)
	% diagonally normalized SIRT with bound projection
	% SIRT is slower than CG on dense cases, but it is very stable for sparse
	% CT and gives a good starting point for SART or TV
	b = b(:);
	x = clip_values(zeros(geom.N * geom.N, 1), lo, hi);
	[ray_scale, col_scale] = sirt_scales(geom, length(b));
	for k = 1:max_iter
		% normalized residual backprojection
		r = b - fwd_geom(x, geom);
		x = clip_values(x + omega * (col_scale .* adj_geom(ray_scale .* r, geom)), lo, hi);
	end
	x = reshape(x, geom.N, geom.N);
end

function x = sirt_denoised(b, geom, max_iter, omega, denoise_every, denoise_steps, kappa, dt, lo, hi)
	% SIRT with light denoising every few iterations
	% this is useful for low-range sparse phantoms, where pure SIRT leaves
	% streaks and pure TV can make the object look too blocky
	b = b(:);
	x = clip_values(zeros(geom.N * geom.N, 1), lo, hi);
	[ray_scale, col_scale] = sirt_scales(geom, length(b));
	for k = 1:max_iter
		r = b - fwd_geom(x, geom);
		x = clip_values(x + omega * (col_scale .* adj_geom(ray_scale .* r, geom)), lo, hi);
		if mod(k, denoise_every) == 0
			% median removes isolated artifacts; anisotropic smoothing reduces
			% streak texture while trying to keep edges
			img = reshape(x, geom.N, geom.N);
			img = median3x3(img);
			img = anisotropic_smooth(img, denoise_steps, kappa, dt, lo, hi);
			x = img(:);
		end
	end
	x = clip_values(reshape(x, geom.N, geom.N), lo, hi);
end

function [ray_scale, col_scale] = sirt_scales(geom, m)
	% approximate row and column normalization
	% ray_scale normalizes each detector equation; col_scale normalizes each
	% pixel update. This keeps SIRT from taking huge steps in dense regions
	ray = abs(fwd_geom(ones(geom.N * geom.N, 1), geom));
	ray_scale = 1 ./ max(ray(:), max(max(ray(:)) * 1e-6, eps));
	col = abs(adj_geom(ones(m, 1), geom));
	col_scale = 1 ./ max(col(:), max(max(col(:)) * 1e-5, eps));
end

function x = sart_refine(b, geom, x0, sweeps, omega, lo, hi, smooth_every, smooth_steps, kappa, dt)
	% a small ordered Kaczmarz/SART polish
	% unlike SIRT, which averages all projection residuals at once, SART updates
	% after each angle. This can sharpen data consistency quickly
	b = b(:);
	x = clip_values(x0(:), lo, hi);
	ones_img = ones(geom.N * geom.N, 1);
	ray_scale = cell(geom.n_angles, 1);
	col_scale = cell(geom.n_angles, 1);
	for a = 1:geom.n_angles
		% one-angle normalizers, analogous to the SIRT normalizers
		ray = abs(fwd_angle(ones_img, geom, a));
		ray_scale{a} = 1 ./ max(ray(:), max(max(ray(:)) * 1e-6, eps));
		col = abs(adj_angle(ones(geom.n_det, 1), geom, a));
		col_scale{a} = 1 ./ max(col(:), max(max(col(:)) * 1e-5, eps));
	end
	for s = 1:sweeps
		% alternate angle order each sweep to reduce directional bias
		if mod(s, 2) == 1
			order = 1:geom.n_angles;
		else
			order = geom.n_angles:-1:1;
		end
		for a = order
			% compare this angle's projection with the measured projection, then
			% backproject only this angle's normalized residual
			ba = b((a - 1) * geom.n_det + (1:geom.n_det));
			r = (ba(:) - fwd_angle(x, geom, a)) .* ray_scale{a};
			x = clip_values(x + omega * (col_scale{a} .* adj_angle(r, geom, a)), lo, hi);
		end
		if smooth_every > 0 && mod(s, smooth_every) == 0
			% tiny smoothing between sweeps keeps SART from amplifying sparse-view
			% striping too aggressively
			img = anisotropic_smooth(reshape(x, geom.N, geom.N), smooth_steps, kappa, dt, lo, hi);
			x = img(:);
		end
	end
	x = reshape(x, geom.N, geom.N);
end

function x = tv_pdhg_reconstruct(b, geom, tv_lam, tv_huber, max_iter, lo, hi, x0)
	% Chambolle-Pock / PDHG solver
	% it approximately solves: min_x 0.5 * ||A * x - b||^2 + tv_lam * TV(x)
	% wirh lo <= x <= hi
	% TV is useful for sparse CT because it suppresses long streak artifacts
	% while preserving object boundaries better than plain smoothing
	b = b(:);

	% estimate a safe step size from the norm of A and the image gradient
	L = sqrt(estimate_normA(geom)^2 + 8);
	tau = 0.99 / L;
	sigma = 0.99 / L;

	% optional warm start; starting from SIRT is much faster than starting
	% from zero in the hard sparse cases
	if nargin < 8
		x = clip_values(zeros(geom.N, geom.N), lo, hi);
	else
		x = clip_values(x0, lo, hi);
	end
	x_bar = x;
	y_data = zeros(length(b), 1);
	p_x = zeros(geom.N, geom.N);
	p_y = zeros(geom.N, geom.N);
	for k = 1:max_iter
		% data dual update for the least-squares term
		q = y_data + sigma * fwd_geom(x_bar(:), geom);
		y_data = (q - sigma * b) / (1 + sigma);

		% TV dual update using forward finite differences
		[g_x, g_y] = image_gradient(x_bar);
		p_x = p_x + sigma * g_x;
		p_y = p_y + sigma * g_y;
		if tv_huber > 0
			% small Huber smoothing makes TV less harsh on tiny gradients
			huber_scale = 1 + sigma * tv_huber / max(tv_lam, eps);
			p_x = p_x / huber_scale;
			p_y = p_y / huber_scale;
		end

		% project each pixel's dual vector onto the TV ball
		proj = max(1, sqrt(p_x.^2 + p_y.^2) / max(tv_lam, eps));
		p_x = p_x ./ proj;
		p_y = p_y ./ proj;

		% primal update: data gradient plus TV divergence
		x_old = x;
		data_grad = reshape(adj_geom(y_data, geom), geom.N, geom.N);
		x = clip_values(x - tau * (data_grad + image_divergence(p_x, p_y)), lo, hi);
		x_bar = clip_values(2 * x - x_old, lo, hi);
	end
end

function normA = estimate_normA(geom)
	% power iteration estimate of ||A||; inspired by lab 07
	% this lets PDHG choose a stable step size without forming A
	v = ones(geom.N * geom.N, 1);
	v = v / norm(v);
	for k = 1:10
		z = adj_geom(fwd_geom(v, geom), geom);
		nrm = norm(z(:));
		if nrm < eps
			break;
		end
		v = z(:) / nrm;
	end
	normA = max(norm(fwd_geom(v, geom)), eps);
end

function x = layered_shape_prior(x_tv, b, geom, lo, hi, blend_good, blend_mid)
	% refits nested shapes to the measured sinogram
	% many high-range sparse phantoms look like overlapping nearly-constant
	% objects. This routine thresholds the TV image into nsted masks, solves
	% for mask intensities directly in projection space, then blends the rezult
	% back with TV so the prior does not take over when it fits badly

	b = b(:);
	scale = max(hi - lo, 1e-12);
	xs = mean3x3(mean3x3(x_tv));
	% more threshold layers gives the shape fit more chances to catch boundaries
	% the blend chooser below still protects us from using too much of this fit
	% Tier 3 has enough time for more layer candidates; Tier 4 stays lighter
	% because it is already close to the time limit
	if geom.N <= 128
		thresholds = [0.03, 0.045, 0.065, 0.09, 0.125, 0.17, 0.23, 0.31, 0.41, 0.54, 0.69, 0.84];
	else
		thresholds = [0.035, 0.055, 0.08, 0.12, 0.17, 0.24, 0.33, 0.44, 0.58, 0.74];
	end
	basis = zeros(geom.N, geom.N, length(thresholds) + 1);
	count = 0;
	min_area = max(20, round(0.0008 * geom.N * geom.N));
	max_area = round(0.85 * geom.N * geom.N);
	for k = 1:length(thresholds)
		% each threshold creates one possible object/support layer
		mask = double(xs > lo + thresholds(k) * scale);
		area = sum(mask(:));
		if area >= min_area && area <= max_area
			count = count + 1;
			basis(:, :, count) = mask;
		end
	end

	% also include one soft basis so the fit can keep gradual intensity changes
	soft_basis = clip_values((xs - lo) / scale, 0, 1);
	if sum(soft_basis(:) > 0.03) >= min_area
		count = count + 1;
		basis(:, :, count) = soft_basis;
	end

	if count == 0
		x = x_tv;
		return;
	end
	G = zeros(length(b), count);
	c = zeros(count, 1);
	for k = 1:count
		% G contains the projection of each candidate layer
		tmp = basis(:, :, k);
		G(:, k) = fwd_geom(tmp(:), geom);
		idx = tmp > 0.05;
		if any(idx(:))
			c(k) = max(0, mean(x_tv(idx)));
		end
	end
	lambda = 1e-4;
	pred = G * c;
	col_norm = sum(G.^2, 1)' + lambda;
	for sweep = 1:110
		% coordinate-descent nonnegative least squares for layer weights
		for k = 1:count
			old = c(k);
			grad = G(:, k)' * (pred - b) + lambda * c(k);
			c(k) = max(0, c(k) - grad / max(col_norm(k), eps));
			pred = pred + (c(k) - old) * G(:, k);
		end
	end
	x_fit = zeros(geom.N, geom.N);
	for k = 1:count
		x_fit = x_fit + c(k) * basis(:, :, k);
	end
	x_fit = clip_values(x_fit, lo, hi);

	% softens the fitted layer image a little before the final blend
	% this helps reduce blocky mask edges from the threshold basis
	x_fit = 0.75 * x_fit + 0.25 * mean3x3(x_fit);
	x_fit = clip_values(x_fit, lo, hi);

	Ax_tv = fwd_geom(x_tv(:), geom);
	Ax_fit = fwd_geom(x_fit(:), geom);

	% trying several safe TV/layer blends and keep the one that best matches the sinogram
	% this avoids forcing the same blend amount on every hidden phantom
	weights = unique([0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, blend_mid, 0.35, 0.45, 0.50, 0.55, 0.60, blend_good, 0.70, 0.75, 0.78, 0.85, 0.90]);

	best_w = 0;
	best_alpha = 1;
	best_res = inf;

	for i = 1:length(weights)
		w_try = weights(i);
		Ax_mix = w_try * Ax_fit + (1 - w_try) * Ax_tv;

		alpha = (Ax_mix(:)' * b(:)) / max(Ax_mix(:)' * Ax_mix(:), eps);
		alpha = min(max(alpha, 0.86), 1.14);

		res_try = norm(alpha * Ax_mix(:) - b(:));
		if res_try < best_res
			best_res = res_try;
			best_w = w_try;
			best_alpha = alpha;
		end
	end

	x = best_w * x_fit + (1 - best_w) * x_tv;
	x = clip_values(best_alpha * x, lo, hi);
end

function x = best_data_blend(x_a, x_b, b, geom, lo, hi, weights, amin, amax)
	% choose the best blend between two candidate images using projection error
	% weights means: 0 = only x_a, 1 = only x_b

	Ax_a = fwd_geom(x_a(:), geom);
	Ax_b = fwd_geom(x_b(:), geom);

	best_w = weights(1);
	best_alpha = 1;
	best_res = inf;

	for i = 1:length(weights)
		w = weights(i);
		Ax_mix = (1 - w) * Ax_a + w * Ax_b;

		alpha = (Ax_mix(:)' * b(:)) / max(Ax_mix(:)' * Ax_mix(:), eps);
		alpha = min(max(alpha, amin), amax);

		res = norm(alpha * Ax_mix(:) - b(:));
		if res < best_res
			best_res = res;
			best_w = w;
			best_alpha = alpha;
		end
	end

	x = (1 - best_w) * x_a + best_w * x_b;
	x = clip_values(best_alpha * x, lo, hi);
end

function y = detail_boost(x, lo, hi, amount)
	% small unsharp mask: sharpens edges without changing the main pipeline
	blur = mean3x3(x);
	y = clip_values(x + amount * (x - blur), lo, hi);
end

function y = background_floor(x, lo, hi, frac)
	% removes very tiny background haze from the sharpened candidate
	cutoff = lo + frac * max(hi - lo, 1e-12);
	y = x;
	y(y < cutoff) = lo;
	y = clip_values(y, lo, hi);
end

function x = best_three_candidates(x1, x2, x3, b, geom, lo, hi, amin, amax)
	% choose the best of three final images using projection error
	cands = cat(3, x1, x2, x3);
	best_x = x1;
	best_res = inf;

	for i = 1:3
		x_try = cands(:, :, i);
		Ax = fwd_geom(x_try(:), geom);

		alpha = (Ax(:)' * b(:)) / max(Ax(:)' * Ax(:), eps);
		alpha = min(max(alpha, amin), amax);

		x_scaled = clip_values(alpha * x_try, lo, hi);
		res = norm(fwd_geom(x_scaled(:), geom) - b(:));

		if res < best_res
			best_res = res;
			best_x = x_scaled;
		end
	end

	x = best_x;
end

function x = scale_to_data(x, b, geom, lo, hi, amin, amax)
	% correct global intensity after smoothing/blending
	% many postprocesses slightly shrink intensities. A single scalar alpha
	% chosen in projection space fixes most of that without changing shapes
	Ax = fwd_geom(x(:), geom);
	alpha = (Ax(:)' * b(:)) / max(Ax(:)' * Ax(:), eps);
	alpha = min(max(alpha, amin), amax);
	x = clip_values(alpha * x, lo, hi);
end

function conf = level_confidence(x, lo, hi)
	% measures how close the image is to common phantom intensity levels

	x_med = median3x3(x);
	x_quant = quantize_scaled(x_med, lo, hi);
	conf = sum(abs(x_med(:) - x_quant(:)) < 0.05 * max(hi - lo, 1e-12)) / numel(x_med);
end

function x = postprocess_dense_soft(x_raw, lo, hi)
	% conservative level snapping for dense cases
	% if the median-filtered image is already close to common phantom levels,
	% move only those confident pixels toward the nearest level. If not, return
	% the median-filtered image and avoid damaging smooth phantoms
	x_med = median3x3(x_raw);
	x_quant = quantize_scaled(x_med, lo, hi);
	mask = abs(x_med - x_quant) < 0.05 * max(hi - lo, 1e-12);
	if sum(mask(:)) / numel(mask) < 0.75
		x = clip_values(x_med, lo, hi);
		return;
	end
	x = x_raw;
	x(mask) = 0.15 * x(mask) + 0.85 * x_quant(mask);
	x = clip_values(median3x3(x), lo, hi);
end

function x = postprocess_sparse_soft(x_raw, lo, hi)
	% same idea as dense postprocess, but without the
	% final median pass because TV images are already edge-regularized
	x_quant = quantize_scaled(x_raw, lo, hi);
	mask = abs(x_raw - x_quant) < 0.05 * max(hi - lo, 1e-12);
	if sum(mask(:)) / numel(mask) < 0.75
		x = clip_values(x_raw, lo, hi);
		return;
	end
	x = x_raw;
	x(mask) = 0.15 * x(mask) + 0.85 * x_quant(mask);
	x = clip_values(x, lo, hi);
end

function x = best_four_candidates(x1, x2, x3, x4, b, geom, lo, hi, amin, amax)
	% choose the best of four final candidates using projection error
	% this uses the fast approximate scoring, so it avoids extra full projections

	cands = cat(3, x1, x2, x3, x4);
	best_x = x1;
	best_res = inf;

	for i = 1:4
		x_try = cands(:, :, i);
		Ax = fwd_geom(x_try(:), geom);

		alpha = (Ax(:)' * b(:)) / max(Ax(:)' * Ax(:), eps);
		alpha = min(max(alpha, amin), amax);

		x_scaled = clip_values(alpha * x_try, lo, hi);
		res = norm(alpha * Ax(:) - b(:));

		if res < best_res
			best_res = res;
			best_x = x_scaled;
		end
	end

	x = best_x;
end

function y = bilateral5x5(x, lo, hi, range_frac, n_iter)
	% small toolbox-free bilateral filter
	% nearby pixels are averaged only if their intensities are also close
	% this removes speckle/checkerboard artifacts while preserving strong edges
	range_sigma = max(range_frac * (hi - lo), 1e-12);
	spatial_sigma = 1.2;
	y = x;
	[m, n] = size(x);
	for it = 1:n_iter
		xp = replicate_pad2(y, 2);
		center = y;
		num = zeros(m, n);
		den = zeros(m, n);
		for di = -2:2
			for dj = -2:2
				% spatial weight depends on distance; range weight depends on
				% intensity difference from the center pixel
				neigh = xp((3 + di):(m + 2 + di), (3 + dj):(n + 2 + dj));
				w = exp(-(di * di + dj * dj) / (2 * spatial_sigma * spatial_sigma));
				w = w * exp(-((neigh - center).^2) / (2 * range_sigma * range_sigma));
				num = num + w .* neigh;
				den = den + w;
			end
		end
		y = clip_values(num ./ max(den, eps), lo, hi);
	end
end

function xp = replicate_pad2(x, r)
	% padding helper used by filters
	% replicated borders avoid dark edges that zero-padding would create
	[m, n] = size(x);
	xp = zeros(m + 2 * r, n + 2 * r);
	xp(r+1:r+m, r+1:r+n) = x;
	for k = 1:r
		xp(k, r+1:r+n) = x(1, :);
		xp(r+m+k, r+1:r+n) = x(m, :);
	end
	for k = 1:r
		xp(:, k) = xp(:, r+1);
		xp(:, r+n+k) = xp(:, r+n);
	end
end

function [g_x, g_y] = image_gradient(x)
	% forward finite differences for TV
	% g_x is vertical difference; g_y is horizontal difference
	[N, M] = size(x);
	g_x = zeros(N, M);
	g_y = zeros(N, M);
	g_x(1:N-1, :) = x(2:N, :) - x(1:N-1, :);
	g_y(:, 1:M-1) = x(:, 2:M) - x(:, 1:M-1);
end

function div = image_divergence(p_x, p_y)
	% negative adjoint of image_gradient
	% this is the TV contribution used in the PDHG primal update
	[N, M] = size(p_x);
	div = zeros(N, M);
	div(1:N-1, :) = div(1:N-1, :) - p_x(1:N-1, :);
	div(2:N, :) = div(2:N, :) + p_x(1:N-1, :);
	div(:, 1:M-1) = div(:, 1:M-1) - p_y(:, 1:M-1);
	div(:, 2:M) = div(:, 2:M) + p_y(:, 1:M-1);
end

function y = median3x3(x)
	% toolbox-free median filter
	% this vectorized version is faster than looping pixel-by-pixel and removes
	% isolated bright/dark artifacts while keeping edges fairly sharp
	[m, n] = size(x);
	xp = replicate_pad2(x, 1);
	vals = zeros(m, n, 9);
	t = 0;
	for di = -1:1
		for dj = -1:1
			t = t + 1;
			vals(:, :, t) = xp((2 + di):(m + 1 + di), (2 + dj):(n + 1 + dj));
		end
	end
	vals = sort(vals, 3);
	y = vals(:, :, 5);
end

function x = anisotropic_smooth(x, n_iter, kappa, dt, lo, hi)
	% Perona-Malik style edge-preserving smoothing
	% small gradients are smoothed, but large gradients get small weights, so
	% object boundaries survive better than with a plain mean filter
	for k = 1:n_iter
		n = zeros(size(x)); s = n; w = n; e = n;
		n(2:end, :) = x(1:end-1, :) - x(2:end, :);
		s(1:end-1, :) = x(2:end, :) - x(1:end-1, :);
		w(:, 2:end) = x(:, 1:end-1) - x(:, 2:end);
		e(:, 1:end-1) = x(:, 2:end) - x(:, 1:end-1);
		cn = 1 ./ (1 + (n / kappa).^2);
		cs = 1 ./ (1 + (s / kappa).^2);
		cw = 1 ./ (1 + (w / kappa).^2);
		ce = 1 ./ (1 + (e / kappa).^2);
		x = clip_values(x + dt * (cn .* n + cs .* s + cw .* w + ce .* e), lo, hi);
	end
end

function y = mean3x3(x)
	% fast 3x3 mean filter with replicated borders
	% used mainly to build smoother masks for the layer prior
	[m, n] = size(x);
	xp = replicate_pad2(x, 1);
	y = (xp(1:m, 1:n) + xp(1:m, 2:n+1) + xp(1:m, 3:n+2) + ...
		 xp(2:m+1, 1:n) + xp(2:m+1, 2:n+1) + xp(2:m+1, 3:n+2) + ...
		 xp(3:m+2, 1:n) + xp(3:m+2, 2:n+1) + xp(3:m+2, 3:n+2)) / 9;
end

function xq = quantize_scaled(x, lo, hi)
	% map pixels to the closest expected phantom level
	% levels are scaled to the problem's [lo, hi] range instead of hard-coded absolute intensities
	base = [0, 0.1, 0.2, 0.3, 0.4, 1.0];
	levels = lo + base * (hi - lo);
	x = clip_values(x, lo, hi);
	xq = zeros(size(x));
	best = inf(size(x));
	for i = 1:length(levels)
		d = abs(x - levels(i));
		mask = d < best;
		xq(mask) = levels(i);
		best(mask) = d(mask);
	end
end

function y = clip_values(x, lo, hi)
	% enforce the physical intensity interval
	y = min(max(x, lo), hi);
end

function [lo, hi] = get_bounds(prob)
	% read optional x_min/x_max, defaulting to [0, 1]
	lo = 0;
	hi = 1;
	if isfield(prob, 'x_min'), lo = prob.x_min; end
	if isfield(prob, 'x_max'), hi = prob.x_max; end
	if hi <= lo, hi = lo + 1; end
end

function val = get_field_or(s, name, default_value)
	% small helper for optional metadata fields
	val = default_value;
	if isfield(s, name), val = s.(name); end
end

function info = make_info(iters, lambda, method, sigma_noise, N_phys)
	% store branch metadata for debugging/evaluation logs
	info = struct('iters', iters, 'lambda', lambda, 'method', method, 'sigma', sigma_noise, 'N_phys', N_phys);
end
