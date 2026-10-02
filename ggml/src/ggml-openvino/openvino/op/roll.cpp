#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <openvino/op/constant.hpp>
#include <openvino/op/roll.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_roll(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    const int32_t * params = context.get_output_op_params();

    auto input = context.get_input(0);
    auto shift = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4}, std::vector<int64_t>{params[3], params[2], params[1], params[0]});
    auto axes = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 1, 2, 3});

    auto result = std::make_shared<ov::op::v7::Roll>(input, shift, axes);
    return rename_outputs_with_suffix({result}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
