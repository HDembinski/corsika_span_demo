#include <Eigen/Core>
#include <array>
#include <benchmark/benchmark.h>
#include <type_traits>
#include <variant>

// must contain only floats and should be a trivial type for performance
struct Particle {
  float& px() { return px_; }
  float& py() { return py_; }
  float& pz() { return pz_; }
  float& e() { return e_; }

  // "private" variables
  float px_, py_, pz_, e_;    
};

static_assert(std::is_trivial<Particle>::value);

class ParticleSpan {
public:
    using iterator = Particle*;
    using ArrayView = Eigen::Map<
        Eigen::Array<float, Eigen::Dynamic, 1>,
        Eigen::Unaligned,
        Eigen::InnerStride<(sizeof(Particle) / sizeof(float))>
    >;

    ParticleSpan(std::vector<Particle>& v) : begin_(v.data()), end_(v.data() + v.size()) {};

    iterator begin() { return begin_; }
    iterator end() { return end_; }

    std::size_t size() { return end_ - begin_; }

    ArrayView px() { return ArrayView(&begin_->px_, size()); }
    ArrayView py() { return ArrayView(&begin_->py_, size()); }
    ArrayView pz() { return ArrayView(&begin_->pz_, size()); }
    ArrayView e() { return ArrayView(&begin_->e_, size()); }

private:
    iterator begin_, end_;
};

template <class T>
decltype(auto) sqr(const T& x) {
  return x * x;
}

template <class T>
decltype(auto) momentum_squared(T& part) {
  return sqr(part.px()) + sqr(part.py()) + sqr(part.pz());
}

// note: this function body looks the same whether we pass one particle or a span!
template <class T>
void do_energy_loss(T& part) {
  auto beta_2 = momentum_squared(part) / sqr(part.e());
  // compute energy loss, ignoring all constants
  using std::log; // allow Eigen to find its own log via ADL
  auto energy_loss = log(beta_2 / (1.0 - beta_2)) / beta_2 - 1.0;
  part.e() -= energy_loss;
}

struct ContinuousEnergyLoss {
  template <class T>
  void operator()(T& p) const { do_energy_loss(p); }
};

struct DummyProcess {
  template <class T>
  void operator()(T& p) const {}
};

// Method 1: process one particle at once
static void process_one(benchmark::State& state) {
  std::vector<Particle> stack(static_cast<unsigned>(state.range(0)));
  ParticleSpan span(stack);

  for (auto _ : state) {
    for (auto&& p : span)
      do_energy_loss(p);    
  }
}

// Method 2: process block of particles at once
static void process_span(benchmark::State& state) {
  std::vector<Particle> stack(static_cast<unsigned>(state.range(0)));
  ParticleSpan span(stack);

  for (auto _ : state) {
    do_energy_loss(span);
  }
}

// Method 1A: process one particle at once using std::variant of processes
static void variant_process_one(benchmark::State& state) {
  std::vector<Particle> stack(static_cast<unsigned>(state.range(0)));
  ParticleSpan span(stack);

  std::variant<ContinuousEnergyLoss, DummyProcess> process;
  process = ContinuousEnergyLoss(); 
  for (auto _ : state) {
    for (auto&& p : span)
      visit([&span](auto& proc) { proc(span); }, process);
  }
}

// Method 2A: process block of particles at once using std::variant of processes
static void variant_process_span(benchmark::State& state) {
  std::vector<Particle> stack(static_cast<unsigned>(state.range(0)));
  ParticleSpan span(stack);

  std::variant<ContinuousEnergyLoss, DummyProcess> process;
  process = ContinuousEnergyLoss(); 
  for (auto _ : state)
    visit([&span](auto& proc) { proc(span); }, process);
}

BENCHMARK(process_one)->RangeMultiplier(10)->Range(10, 10000);
BENCHMARK(process_span)->RangeMultiplier(10)->Range(10, 10000);
BENCHMARK(variant_process_one)->RangeMultiplier(10)->Range(10, 10000);
BENCHMARK(variant_process_span)->RangeMultiplier(10)->Range(10, 10000);