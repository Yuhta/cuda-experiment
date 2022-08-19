#include "IndicesOfSetBits.h"

#include <cstddef>

namespace {

template <typename T, typename U>
constexpr T roundUp(T value, U factor) {
  return (value + (factor - 1)) / factor * factor;
}

constexpr uint64_t lowMask(int32_t bits) {
  return (1UL << bits) - 1;
}

constexpr uint64_t highMask(int32_t bits) {
  return lowMask(bits) << (64 - bits);
}

constexpr inline uint64_t nbytes(int32_t bits) {
  return roundUp(bits, 8) / 8;
}

template <typename PartialWordFunc, typename FullWordFunc>
inline void forEachWord(
    int32_t begin,
    int32_t end,
    PartialWordFunc partialWordFunc,
    FullWordFunc fullWordFunc) {
  if (begin >= end) {
    return;
  }
  int32_t firstWord = roundUp(begin, 64);
  int32_t lastWord = end & ~63L;
  if (lastWord < firstWord) {
    partialWordFunc(
        lastWord / 64, lowMask(end - lastWord) & highMask(firstWord - begin));
    return;
  }
  if (begin != firstWord) {
    partialWordFunc(begin / 64, highMask(firstWord - begin));
  }
  for (int32_t i = firstWord; i + 64 <= lastWord; i += 64) {
    fullWordFunc(i / 64);
  }
  if (end != lastWord) {
    partialWordFunc(lastWord / 64, lowMask(end - lastWord));
  }
}

template <typename Callable>
void forEachBit(
    const uint64_t* bits,
    int32_t begin,
    int32_t end,
    bool isSet,
    Callable func) {
  static constexpr uint64_t kAllSet = -1ULL;
  forEachWord(
      begin,
      end,
      [isSet, bits, func](int32_t idx, uint64_t mask) {
        auto word = (isSet ? bits[idx] : ~bits[idx]) & mask;
        if (!word) {
          return;
        }
        while (word) {
          func(idx * 64 + __builtin_ctzll(word));
          word &= word - 1;
        }
      },
      [isSet, bits, func](int32_t idx) {
        auto word = (isSet ? bits[idx] : ~bits[idx]);
        if (kAllSet == word) {
          const size_t start = idx * 64;
          const size_t end = (idx + 1) * 64;
          for (size_t row = start; row < end; ++row) {
            func(row);
          }
        } else {
          while (word) {
            func(idx * 64 + __builtin_ctzll(word));
            word &= word - 1;
          }
        }
      });
}

template <typename Callable>
inline void
forEachSetBit(const uint64_t* bits, int32_t begin, int32_t end, Callable func) {
  forEachBit(bits, begin, end, true, func);
}

}

void indicesOfSetBits(const uint64_t* bits, int n, int& outCount, int32_t* out) {
  outCount = 0;
  forEachSetBit(bits, 0, n, [&](int32_t i) { out[outCount++] = i; });
}
