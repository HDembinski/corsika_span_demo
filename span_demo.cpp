#include <Eigen/Core>
#include <benchmark/benchmark.h>
#include <type_traits>
#include <variant>
#include <vector>
#include <cstdint>
#include <random>

// must have size divisible by size of float and should be trivial for performance
struct Particle {
  std::int32_t& pid() { return pid_; }

  float& px() { return px_; }
  float& py() { return py_; }
  float& pz() { return pz_; }
  float& e() { return e_; }

  float& x() { return x_; }
  float& y() { return y_; }
  float& z() { return z_; }
  float& t() { return t_; }

  // "private" variables
  std::int32_t pid_;
  float px_, py_, pz_, e_;
  float x_, y_, z_, t_;
};

// quantity does not have a trivial default constructor, this may be a performance issue
static_assert(std::is_trivial<Particle>::value);

// size of Particle must be multiple of size of float for Eigen::Map to work
static_assert(sizeof(Particle) % sizeof(float) == 0);

template <class T>
using ArrayView = Eigen::Map<
    Eigen::Array<T, Eigen::Dynamic, 1>,
    Eigen::Unaligned,
    Eigen::InnerStride<(sizeof(Particle) / sizeof(float))>
>;

class ParticleSpan {
public:
    using pointer = Particle*;
    using iterator = pointer;
    using ArrayFView = ArrayView<float>;
    using ArrayIView = ArrayView<std::int32_t>;

    ParticleSpan(pointer b, pointer e) : begin_(b), end_(e) {};

    iterator begin() { return begin_; }
    iterator end() { return end_; }

    std::size_t size() { return end_ - begin_; }

    ArrayIView pid() { return ArrayIView(&begin_->pid_, size()); }
    ArrayFView px() { return ArrayFView(&begin_->px_, size()); }
    ArrayFView py() { return ArrayFView(&begin_->py_, size()); }
    ArrayFView pz() { return ArrayFView(&begin_->pz_, size()); }
    ArrayFView e() { return ArrayFView(&begin_->e_, size()); }
    ArrayFView x() { return ArrayFView(&begin_->x_, size()); }
    ArrayFView y() { return ArrayFView(&begin_->y_, size()); }
    ArrayFView z() { return ArrayFView(&begin_->z_, size()); }
    ArrayFView t() { return ArrayFView(&begin_->t_, size()); }

private:
    iterator begin_, end_;
};

// setup the particle stack for the benchmarks
auto setup_stack() {
  std::vector<Particle> stack(100000);
  int i = 0;
  // make 1/3 of particles neutral
  for (auto&& part : stack)
    part.pid() = (++i % 3 - 1);
  return stack;
}

template <class T>
decltype(auto) sqr(const T& x) {
  return x * x;
}

template <class T>
decltype(auto) momentum_squared(T& part) {
  return sqr(part.px()) + sqr(part.py()) + sqr(part.pz());
}

template <class T>
decltype(auto) charge(T& part) {
  // this could be a complicated function or a lookup table
  return part.pid() != 0;
}

decltype(auto) charge(ParticleSpan& part) {
  // this could be a complicated function or a lookup table
  return (part.pid() != 0).template cast<float>();
}

// note: this function body looks the same whether we pass one particle or a span!
template <class T>
void energy_loss(T& part) {
  decltype(auto) beta_2 = momentum_squared(part) / sqr(part.e());
  // compute energy loss, ignoring all constants
  using std::log; // allow Eigen to find its own log via ADL
  decltype(auto) energy_loss = charge(part) * (log(beta_2 / (1.0 - beta_2)) / beta_2 - 1.0);
  part.e() -= energy_loss;
}

// ... nevertheless we try a special one particle version for comparison
void energy_loss(Particle& part) {
  auto beta_2 = momentum_squared(part) / sqr(part.e());
  const auto c = charge(part);
  if (c != 0) {
    // compute energy loss, ignoring all constants
    const auto energy_loss = c * (std::log(beta_2 / (1.0 - beta_2)) / beta_2 - 1.0);
    part.e() -= energy_loss;    
  }
}

// note: again function body looks the same whether we pass one particle or a span
template <class T>
void move_particle(T& p) {
  const auto dt = 0.1;
  p.x() += p.px() * dt;
  p.y() += p.py() * dt;
  p.z() += p.pz() * dt;
  p.t() += dt;
}

struct ContinuousEnergyLoss {
  template <class T>
  void operator()(T& p) const { energy_loss(p); }
};

struct ContinuousEnergyLossNoEigen {
  void operator()(ParticleSpan& span) const {
    for (auto&& p : span)
      energy_loss(p);
  }
};

struct MoveParticle {
  template <class T>
  void operator()(T& p) const { move_particle(p); }
};

struct MoveParticleNoEigen {
  void operator()(ParticleSpan& span) const {
    for (auto&& p : span)
      move_particle(p);
  }
};

using ProcessVariant = std::variant<ContinuousEnergyLoss, ContinuousEnergyLossNoEigen,
                                    MoveParticle, MoveParticleNoEigen>;
using ProcessList = std::vector<ProcessVariant>;

// Method 1: process one particle at once
static void process_one(benchmark::State& state) {
  auto stack = setup_stack();

  ParticleSpan span(stack.data(), stack.data() + state.range(0));

  for (auto _ : state) {
    for (auto&& p : span) {
      energy_loss(p);   
      move_particle(p);
    }
  }
}

// Method 2: process block of particles at once
static void process_span(benchmark::State& state) {
  auto stack = setup_stack();

  ParticleSpan span(stack.data(), stack.data() + state.range(0));

  for (auto _ : state) {
    energy_loss(span);
    move_particle(span);
  }
}

// Method 3: like Method 2, but don't use Eigen
static void process_span_no_eigen(benchmark::State& state) {
  auto stack = setup_stack();

  ParticleSpan span(stack.data(), stack.data() + state.range(0));

  for (auto _ : state) {
    for (auto&& p : span) {
      energy_loss(p);
      move_particle(p);
    }
  }
}

// Method 1A: process one particle at once using std::variant of processes
static void variant_process_one(benchmark::State& state) {
  auto stack = setup_stack();

  ParticleSpan span(stack.data(), stack.data() + state.range(0));

  ProcessList process_list;
  process_list.emplace_back(ContinuousEnergyLoss());
  process_list.emplace_back(MoveParticle());

  for (auto _ : state) {
    for (auto&& p : span) {
      for (const auto& process : process_list) {
        visit([&span](auto& proc) { proc(span); }, process);
      }
    }
  }
}

// Method 2A: process block of particles at once using std::variant of processes
static void variant_process_span(benchmark::State& state) {
  auto stack = setup_stack();

  ParticleSpan span(stack.data(), stack.data() + state.range(0));

  ProcessList process_list;
  process_list.emplace_back(ContinuousEnergyLoss());
  process_list.emplace_back(MoveParticle());

  for (auto _ : state)
    for (const auto& process : process_list)
      visit([&span](auto& proc) { proc(span); }, process);
}

// Method 3A: like Method 2A, but don't use eigen
static void variant_process_span_no_eigen(benchmark::State& state) {
  auto stack = setup_stack();

  ParticleSpan span(stack.data(), stack.data() + state.range(0));

  ProcessList process_list;
  process_list.emplace_back(ContinuousEnergyLossNoEigen());
  process_list.emplace_back(MoveParticleNoEigen());

  for (auto _ : state)
    for (const auto& process : process_list)
      visit([&span](auto& proc) { proc(span); }, process);
}

BENCHMARK(process_one)->RangeMultiplier(2)->Range(1, 10000);
BENCHMARK(process_span)->RangeMultiplier(2)->Range(1, 10000);
BENCHMARK(process_span_no_eigen)->RangeMultiplier(2)->Range(1, 10000);
BENCHMARK(variant_process_one)->RangeMultiplier(2)->Range(1, 10000);
BENCHMARK(variant_process_span)->RangeMultiplier(2)->Range(1, 10000);
BENCHMARK(variant_process_span_no_eigen)->RangeMultiplier(2)->Range(1, 10000);