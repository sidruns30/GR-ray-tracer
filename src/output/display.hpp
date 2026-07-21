#pragma once

#include <chrono>

#include "../utils.hpp"


namespace Colors
{
    const std::string red = "\033[1;31m";
    const std::string green = "\033[1;32m";
    const std::string yellow = "\033[1;33m";
    const std::string blue = "\033[1;34m";
    const std::string magenta = "\033[1;35m";
    const std::string cyan = "\033[1;36m";
    const std::string white = "\033[1;37m";
    const std::string hotpink = "\033[1;95m";
    const std::string reset = "\033[0m";
}
struct Timer
{
    Timer(const std::string& label)
        : label(label),
          time_begin(std::chrono::steady_clock::now()),
          time_end(time_begin),
          total_time_elapsed(0.) {}
    const std::string& GetLabel() const
    {
        return label;
    }
    void Begin()
    {
        Kokkos::fence();
        time_begin = std::chrono::steady_clock::now();
    }
    void End()
    {
        Kokkos::fence();
        time_end = std::chrono::steady_clock::now();
        total_time_elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_begin).count();
    }
    std::string label;
    std::chrono::time_point<std::chrono::steady_clock> time_begin;
    decltype(std::chrono::steady_clock::now()) time_end;
    double total_time_elapsed;
};

class Timers
{
    public:
        Timers(std::size_t n_iterations,
               std::size_t display_every) :
               n_iterations(n_iterations),
               display_every(display_every) {}
            
        bool TimerExists(const std::string& label) const
        {
            for (const auto& timer : timers)
            {
                if (timer.GetLabel() == label)
                {   return true;}
            }
            return false;
        }

        void AddTimer(const std::string& label)
        {   
            if (!TimerExists(label))
            {   timers.push_back(Timer(label)); }
        }

        void AddTimer(const std::vector<std::string>& labels)
        {
            for (const auto& label : labels)
            {   
                if (!TimerExists(label))
                {   timers.push_back(Timer(label)); }
            }
        }

        void BeginTimer(const std::string& label)
        {
            for (auto& timer : timers)
            {
                if (timer.GetLabel() == label)
                {   timer.Begin(); }
            }
        }

        void EndTimer(const std::string& label)
        {
            for (auto& timer : timers)
            {
                if (timer.GetLabel() == label)
                {   timer.End(); }
            }
        }

        bool PrintTimers(std::size_t current_iteration)
        {
            if (current_iteration % display_every != 0)
            {   return false;}

            double total = 0.;
            for (auto& timer : timers)
            {
                total += timer.total_time_elapsed;
            }
            const std::size_t completed_iterations = current_iteration + 1;
            std::cout << Colors::red << "Iteration: " << Colors::yellow <<
            "[" << completed_iterations << "/" << n_iterations <<
            "]" << Colors::reset << std::endl;

            for (auto& timer : timers)
            {
                const double percent = total > 0.0 ? 100. * timer.total_time_elapsed / total : 0.0;
                std::cout << Colors::blue << timer.GetLabel() << Colors::reset <<
                "[% total]: " << Colors::green << percent << "% \t"
                << Colors::reset << "[time]: " << Colors::yellow <<
                timer.total_time_elapsed * 1.e-9 << " seconds" << std::endl;
            }
            const std::size_t remaining_iterations = n_iterations > completed_iterations
                ? n_iterations - completed_iterations : 0;
            const double estimated_remaining = total * static_cast<double>(remaining_iterations) /
                                               static_cast<double>(completed_iterations);
            std::cout << Colors::red << "Time elapsed: " << Colors::green << total * 1.e-9 << " seconds" << Colors::red;
            std::cout << "\t \t Time left: " << Colors::green << estimated_remaining * 1.e-9
            << " seconds" << Colors::reset << std::endl;
            std::cout << std::endl;
            return true;
        }

        void PrintString(const std::string& str, std::size_t current_iteration)
        {
            if (current_iteration % display_every == 0)
            {   std::cout << Colors::red << str << Colors::reset << std::endl;}
        }

    private:
        std::vector<Timer> timers;
        std::size_t n_iterations;
        std::size_t display_every;
};

inline void WARN(const std::string& message) {
    std::cout << Colors::red << "WARNING: " << message << Colors::reset << std::endl;
}

inline void ERROR(const std::string& message) {
    std::cout << Colors::red << "ERROR: " << message << Colors::reset << std::endl;
}

inline void INFO(const std::string& message) {
    std::cout << Colors::green << "INFO: " << message << Colors::reset << std::endl;
}

// Add color
inline void INFO(const std::string& message, const std::string& color) {
    std::cout << color << "INFO: " << message << Colors::reset << std::endl;
}

inline void INIT(const std::string& message) {
    std::cout << Colors::hotpink << "INIT: " << message << Colors::reset << std::endl;
}
