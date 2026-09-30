#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <openvino/op/constant.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/sigmoid.hpp>
#include <openvino/op/slice.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_glu_geglu_quick(const NodeContext & context) {
    num_inputs_check(context, 1, 2);

    ov::Output<ov::Node> gate;
    ov::Output<ov::Node> up;
    if (context.get_input_size() == 2) {
        gate = process_view_input_new(context, 0);
        up = process_view_input_new(context, 1);
    } else {
        auto combined = process_view_input_new(context, 0);
        const auto combined_shape = combined.get_partial_shape();
        const int64_t last_dim = combined_shape[combined_shape.rank().get_length() - 1].get_length();
        const int64_t half_dim = last_dim / 2;

        auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto gate_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto gate_stop = ov::op::v0::Constant::create(ov::element::i64, {1}, {half_dim});
        auto up_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {half_dim});
        auto up_stop = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * half_dim});

        gate = std::make_shared<ov::op::v8::Slice>(combined, gate_start, gate_stop, step, axis);
        up = std::make_shared<ov::op::v8::Slice>(combined, up_start, up_stop, step, axis);
    }

    const int32_t * params = context.get_output_op_params();
    if (params[1]) {
        std::swap(gate, up);
    }

    const auto input_type = gate.get_element_type();
    auto coefficient = ov::op::v0::Constant::create(input_type, ov::Shape{}, {1.702f});
    auto scaled = std::make_shared<ov::op::v1::Multiply>(gate, coefficient);
    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(scaled);
    auto quick_gelu = std::make_shared<ov::op::v1::Multiply>(gate, sigmoid);
    auto result = std::make_shared<ov::op::v1::Multiply>(quick_gelu, up);

    return rename_outputs_with_suffix({result}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
